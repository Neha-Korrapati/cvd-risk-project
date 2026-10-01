"""FHIR R4 representation of a heart.csv patient, and the reverse mapping from
FHIR resources back to the model's raw inputs.

The paper (Sec. IV-B) states that patient data are exchanged as FHIR resources
but gives no resource types or codes. The mapping below is ours
(DEVIATIONS.md A-16):

* Patient.gender            <- Sex
* Patient.birthDate         <- synthetic, derived from Age (dataset has no dates)
* Observation per feature   <- the other 10 features
  - LOINC where a standard code clearly exists (systolic BP, total cholesterol)
  - a local CodeSystem for exercise-test / ECG findings that have no single
    widely used LOINC code in this form. In a real deployment these would be
    replaced by the hospital's own coded concepts.
"""
from datetime import date

LOINC = "http://loinc.org"
LOCAL = "http://cvd-risk-project.local/fhir/CodeSystem/cvd-features"
UCUM = "http://unitsofmeasure.org"
OBS_CATEGORY = "http://terminology.hl7.org/CodeSystem/observation-category"

# feature -> (system, code, display, kind, unit)
# kind: "quantity" (valueQuantity) | "code" (valueCodeableConcept) | "boolean"
OBSERVATIONS = {
    "RestingBP": (LOINC, "8480-6", "Systolic blood pressure", "quantity", "mm[Hg]"),
    "Cholesterol": (LOINC, "2093-3", "Cholesterol [Mass/volume] in Serum or Plasma", "quantity", "mg/dL"),
    "MaxHR": (LOCAL, "max-heart-rate-exercise", "Maximum heart rate achieved during exercise stress test", "quantity", "/min"),
    "Oldpeak": (LOCAL, "oldpeak", "ST depression relative to previous ECG (Oldpeak)", "quantity", "mV"),
    "FastingBS": (LOCAL, "fasting-blood-sugar-gt-120", "Fasting blood sugar > 120 mg/dL", "boolean", None),
    "ChestPainType": (LOCAL, "chest-pain-type", "Chest pain type", "code", None),
    "RestingECG": (LOCAL, "resting-ecg", "Resting ECG result", "code", None),
    "ST_Slope": (LOCAL, "st-slope-exercise", "Slope of ST segment during exercise stress test", "code", None),
    "ExerciseAngina": (LOCAL, "exercise-induced-angina", "Exercise-induced angina", "boolean", None),
}

CODE_DISPLAY = {
    "ChestPainType": {"TA": "Typical angina", "ATA": "Atypical angina",
                      "NAP": "Non-anginal pain", "ASY": "Asymptomatic"},
    "RestingECG": {"Normal": "Normal", "ST": "ST-T wave abnormality",
                   "LVH": "Left ventricular hypertrophy"},
    "ST_Slope": {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"},
}

# Reference year for synthetic birth dates, fixed so bundles are reproducible.
REFERENCE_YEAR = 2024


def patient_id(row_id) -> str:
    return f"cvd-{row_id}"


def patient_resource(row_id, row) -> dict:
    return {
        "resourceType": "Patient",
        "id": patient_id(row_id),
        "identifier": [{"system": "http://cvd-risk-project.local/heart-csv-row", "value": str(row_id)}],
        "name": [{"use": "official", "family": "Patient", "given": [f"HD-{row_id}"]}],
        "gender": "male" if row["Sex"] == "M" else "female",
        "birthDate": date(REFERENCE_YEAR - int(row["Age"]), 1, 1).isoformat(),
    }


def observation_resource(row_id, feature: str, value) -> dict:
    system, code, display, kind, unit = OBSERVATIONS[feature]
    obs = {
        "resourceType": "Observation",
        "id": f"{patient_id(row_id)}-{code}",
        "status": "final",
        "category": [{"coding": [{"system": OBS_CATEGORY,
                                  "code": "vital-signs" if feature == "RestingBP" else
                                          "laboratory" if feature in ("Cholesterol", "FastingBS") else
                                          "exam"}]}],
        "code": {"coding": [{"system": system, "code": code, "display": display}], "text": display},
        "subject": {"reference": f"Patient/{patient_id(row_id)}"},
    }
    if kind == "quantity":
        obs["valueQuantity"] = {"value": float(value), "unit": unit, "system": UCUM, "code": unit}
    elif kind == "boolean":
        obs["valueBoolean"] = bool(value == "Y" or value == 1)
    else:
        obs["valueCodeableConcept"] = {
            "coding": [{"system": f"{LOCAL}-{feature}", "code": str(value),
                        "display": CODE_DISPLAY[feature][value]}],
            "text": CODE_DISPLAY[feature][value]}
    return obs


def transaction_bundle(raw_rows) -> dict:
    """raw_rows: iterable of (row_id, row) with the original heart.csv columns."""
    entries = []
    for row_id, row in raw_rows:
        resources = [patient_resource(row_id, row)] + [
            observation_resource(row_id, f, row[f]) for f in OBSERVATIONS]
        for r in resources:
            entries.append({"fullUrl": f"{r['resourceType']}/{r['id']}", "resource": r,
                            "request": {"method": "PUT", "url": f"{r['resourceType']}/{r['id']}"}})
    return {"resourceType": "Bundle", "type": "transaction", "entry": entries}


def _code_of(obs: dict) -> tuple[str, str]:
    c = obs["code"]["coding"][0]
    return c["system"], c["code"]


def patient_inputs(patient: dict, observations: list[dict]) -> dict:
    """Map a Patient and its Observations back to raw model inputs.

    Returns {"Sex", "ST_Slope", "ExerciseAngina", "Oldpeak", "ChestPainType"}
    plus any other features found. Missing values are left out, so the caller
    can ask the clinician to fill them in manually.
    """
    by_code = {(s, c): f for f, (s, c, *_rest) in OBSERVATIONS.items()}
    out = {}
    if patient.get("gender") in ("male", "female"):
        out["Sex"] = "M" if patient["gender"] == "male" else "F"
    for obs in observations:
        if obs.get("status") not in (None, "final", "amended", "corrected"):
            continue
        feature = by_code.get(_code_of(obs))
        if feature is None:
            continue
        kind = OBSERVATIONS[feature][3]
        if kind == "quantity":
            out[feature] = float(obs["valueQuantity"]["value"])
        elif kind == "boolean":
            v = bool(obs["valueBoolean"])
            out[feature] = ("Y" if v else "N") if feature == "ExerciseAngina" else int(v)
        else:
            out[feature] = obs["valueCodeableConcept"]["coding"][0]["code"]
    return out
