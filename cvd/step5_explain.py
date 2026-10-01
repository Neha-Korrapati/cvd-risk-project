"""Step 5: explainable AI (Sec. III-C-2, Fig. 4, Appendix E).

(a) SHAP beeswarm of the final model over the hold-out test patients.
(b), (c) SHAP waterfalls of one low-risk and one high-risk test patient. We
pick the test patients whose predicted risk is closest to the paper's two
examples (16.9% and 93.7%).

Run from the project root:
    .venv\\Scripts\\python -m cvd.step5_explain
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import config as C
from . import data
from .predict import RiskModel

PAPER_EXAMPLES = {"low": 0.169, "high": 0.937}


def main() -> None:
    rm = RiskModel()
    X_te, y_te = data.load_split("test")
    raw = data.load_raw()
    Xf = X_te[rm.features]

    print(f"Explaining {len(Xf)} test patients with KernelSHAP...", flush=True)
    sv = rm.shap_values(Xf)
    base = float(rm.explainer.expected_value)
    risk = rm._f(Xf.values)
    print(f"Baseline E[f(x)] = {base:.4f} (paper: 0.555)")
    # Local accuracy check: baseline + sum of SHAP values = predicted risk.
    print(f"Max |baseline + sum(SHAP) - risk| = {np.abs(base + sv.sum(1) - risk).max():.2e}")

    # Show Oldpeak in mV (raw) on the plots; the model itself sees the z-score.
    display = Xf.copy()
    display["Oldpeak"] = raw.loc[Xf.index, "Oldpeak"].values
    exp_all = shap.Explanation(values=sv, base_values=np.full(len(Xf), base),
                               data=display.values, feature_names=rm.features)

    shap.plots.beeswarm(exp_all, show=False, max_display=5)
    plt.title("Feature Effect on CVD Risk for Individual Patients")
    plt.xlabel("SHAP Value (Impact on Probability of CVD)")
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "fig4a_beeswarm.png", dpi=200)
    plt.close()

    rows = []
    for tag, target in PAPER_EXAMPLES.items():
        i = int(np.argmin(np.abs(risk - target)))
        shap.plots.waterfall(exp_all[i], show=False, max_display=5)
        plt.title(f"{'Low' if tag == 'low' else 'High'}-Risk Patient (CVD Risk = {100 * risk[i]:.1f}%)")
        plt.tight_layout()
        plt.savefig(C.FIGURES_DIR / f"fig4{'b' if tag == 'low' else 'c'}_waterfall_{tag}_risk.png", dpi=200)
        plt.close()
        rid = Xf.index[i]
        rows.append({"example": tag, "row_id": rid, "risk": risk[i], "actual": int(y_te.iloc[i]),
                     **{f"raw_{c}": raw.loc[rid, c] for c in
                        ["ST_Slope", "ExerciseAngina", "Sex", "Oldpeak", "ChestPainType"]},
                     **{f"shap_{f}": sv[i, j] for j, f in enumerate(rm.features)}})

    table = pd.DataFrame(rows)
    table.to_csv(C.TABLES_DIR / "fig4_example_patients.csv", index=False)
    mean_abs = pd.Series(np.abs(sv).mean(0), index=rm.features).sort_values(ascending=False)
    mean_abs.to_csv(C.TABLES_DIR / "fig4_mean_abs_shap.csv", header=["mean_abs_shap"])
    direction = pd.Series([np.corrcoef(Xf[f], sv[:, j])[0, 1] for j, f in enumerate(rm.features)],
                          index=rm.features)

    pd.set_option("display.width", 250)
    print("\nExample patients (Fig. 4b, 4c):")
    print(table.round(4).to_string(index=False))
    print("\nMean |SHAP| over test patients:")
    print(mean_abs.round(4).to_string())
    print("\nDirection (corr. of feature value with its SHAP value; + raises risk):")
    print(direction.round(3).to_string())


if __name__ == "__main__":
    main()
