"""Risk prediction + KernelSHAP explanation for one patient, shared by the
Fig. 4 script and the desktop app.

Inputs are raw clinical values (Oldpeak in mV, as recorded in the dataset);
Oldpeak is standardized internally with the scaler fitted in Step 1
(DEVIATIONS.md P-03).
"""
import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import shap

from . import config as C
from . import data


@dataclass
class Explanation:
    risk: float                     # P(CVD) for this patient
    baseline: float                 # E[f(x)], average risk over background patients
    contributions: dict[str, float]  # SHAP value per model feature, probability units
    features: dict[str, float]      # model inputs (Oldpeak standardized)
    high_risk: bool                 # risk >= chosen threshold


def encode_patient(st_slope: str, exercise_angina: str, sex: str, oldpeak_mv: float,
                   chest_pain_type: str, scaler) -> dict[str, float]:
    """Raw clinical values -> the five model features, named as in Table III."""
    numeric = pd.DataFrame([[0, 0, 0, 0, oldpeak_mv]], columns=C.NUMERIC)
    z_oldpeak = float(scaler.transform(numeric)[0, C.NUMERIC.index("Oldpeak")])
    return {
        "ST_Slope_Flat": float(st_slope == "Flat"),
        "ExerciseAngina_Y": float(exercise_angina == "Y"),
        "Sex_M": float(sex == "M"),
        "Oldpeak": z_oldpeak,
        "ChestPainType_ATA": float(chest_pain_type == "ATA"),
    }


class RiskModel:
    def __init__(self):
        self.meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
        self.features = self.meta["features"]
        self.threshold = float(self.meta["threshold"])
        self.model = joblib.load(C.MODELS_DIR / "final_xgb.joblib")
        # Predict on all cores: KernelSHAP needs tens of thousands of predictions
        # per patient. The fitted trees, and so every prediction, are unchanged.
        self.model.set_params(n_jobs=-1)
        self.scaler = joblib.load(C.MODELS_DIR / "scaler.joblib")
        # Background = every training patient, so the SHAP baseline E[f(x)] is the
        # average predicted risk across patients, as in Fig. 4 (ASSUMPTION A-15).
        X_tr, _ = data.load_split("train")
        self.background = X_tr[self.features]
        self.explainer = shap.KernelExplainer(self._f, self.background)
        self.curve = pd.read_csv(C.MODELS_DIR / "test_threshold_curve.csv")

    def _f(self, a: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(pd.DataFrame(a, columns=self.features))[:, 1]

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        # 5 features -> 2^5 = 32 coalitions; nsamples covers all of them, so
        # KernelSHAP is exact here.
        return np.asarray(self.explainer.shap_values(X[self.features], nsamples=64, silent=True))

    def explain(self, x: dict[str, float]) -> Explanation:
        X = pd.DataFrame([x])[self.features]
        risk = float(self._f(X.values)[0])
        sv = self.shap_values(X)[0]
        return Explanation(risk=risk, baseline=float(self.explainer.expected_value),
                           contributions=dict(zip(self.features, map(float, sv))),
                           features={k: float(v) for k, v in x.items()},
                           high_risk=risk >= self.threshold)

    def explain_raw(self, st_slope, exercise_angina, sex, oldpeak_mv, chest_pain_type) -> Explanation:
        return self.explain(encode_patient(st_slope, exercise_angina, sex, oldpeak_mv,
                                           chest_pain_type, self.scaler))
