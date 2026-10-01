"""Step 6: convert the hold-out test patients to FHIR R4 and load them into the
FHIR server (Sec. IV-B).

Only the 20% test split is loaded, so every patient the clinician opens in the
app is one the model never saw during training.

Run from the project root (with the FHIR server running):
    .venv\\Scripts\\python -m cvd.step6_fhir [--base http://127.0.0.1:8080/fhir]
"""
import argparse
import json

import pandas as pd

from . import config as C
from . import data, fhir_mapping
from .fhir_client import DEFAULT_BASE, FhirClient

BUNDLE_PATH = C.ROOT / "fhir" / "test_patients_bundle.json"


def build_bundle() -> dict:
    raw = data.load_raw()
    test_ids = pd.read_csv(C.PROCESSED_DIR / "test.csv", usecols=["row_id"])["row_id"]
    bundle = fhir_mapping.transaction_bundle((rid, raw.loc[rid]) for rid in test_ids)
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUNDLE_PATH.write_text(json.dumps(bundle, indent=1))
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    bundle = build_bundle()
    n_pat = sum(e["resource"]["resourceType"] == "Patient" for e in bundle["entry"])
    print(f"Bundle written: {BUNDLE_PATH} ({n_pat} patients, {len(bundle['entry'])} resources)")
    if args.no_upload:
        return

    client = FhirClient(args.base)
    print(f"FHIR server {args.base}: version {client.ping()}")
    resp = client.transaction(bundle)
    print(f"Uploaded {len(resp['entry'])} resources")

    # Round-trip check: every test patient read back over FHIR must give the
    # same model inputs as the CSV row it came from.
    raw = data.load_raw()
    mismatches = 0
    for e in bundle["entry"]:
        if e["resource"]["resourceType"] != "Patient":
            continue
        pid = e["resource"]["id"]
        rid = int(pid.split("-")[1])
        inputs, _, _ = client.patient_inputs(pid)
        row = raw.loc[rid]
        for k, v in inputs.items():
            if (float(v) != float(row[k])) if isinstance(v, float) else (v != row[k]):
                mismatches += 1
                print(f"  mismatch {pid} {k}: FHIR {v!r} vs CSV {row[k]!r}")
    print(f"Round-trip check: {n_pat} patients, {mismatches} mismatching values")


if __name__ == "__main__":
    main()
