"""Step 4: final XGBoost model on the paper's five features (Sec. II-E, II-F,
III-C, Fig. 3).

Run from the project root:
    .venv\\Scripts\\python -m cvd.step4_final_model
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_predict, cross_validate

from . import config as C
from . import data, stats
from .step2_nested_cv import cv, tune

# Sec. II-F: 100 evenly spaced probability thresholds between 0% and 100%.
THRESHOLDS = np.linspace(0, 1, 100)
# Fig. 3a metrics. ASSUMPTION A-14: the paper's precision / recall / F1 (all 86%
# on the test set) match support-weighted averages at the default 0.5 threshold,
# not positive-class values at 59%; we use the weighted versions for Fig. 3a.
FIG3A_SCORING = {"AUROC": "roc_auc", "Precision": "precision_weighted",
                 "Recall": "recall_weighted", "F1 Score": "f1_weighted"}


def sens_spec(y_true, prob, t: float) -> tuple[float, float, float]:
    pred = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return sens, spec, matthews_corrcoef(y_true, pred)


def curve(y_true, prob) -> pd.DataFrame:
    rows = [(t, *sens_spec(y_true, prob, t)) for t in THRESHOLDS]
    return pd.DataFrame(rows, columns=["threshold", "sensitivity", "specificity", "mcc"])


def metrics_at(y_true, prob, t: float) -> dict:
    pred = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(t),
        "AUROC": roc_auc_score(y_true, prob),
        "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
        "false_negative_rate": fn / (tp + fn), "false_positive_rate": fp / (tn + fp),
        "MCC": matthews_corrcoef(y_true, pred),
        "precision_positive": precision_score(y_true, pred),
        "recall_positive": recall_score(y_true, pred),
        "f1_positive": f1_score(y_true, pred),
        "precision_weighted": precision_score(y_true, pred, average="weighted"),
        "recall_weighted": recall_score(y_true, pred, average="weighted"),
        "f1_weighted": f1_score(y_true, pred, average="weighted"),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def fig3(full_cv, red_cv, y_test, prob, test_curve, t_opt, m_opt) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) full vs reduced, 10-fold CV, Mann-Whitney U
    ax = axes[0, 0]
    rows, pvals = [], {}
    for label in FIG3A_SCORING:
        a, b = full_cv[f"test_{label}"], red_cv[f"test_{label}"]
        rows += [(label, "Full", v) for v in a] + [(label, "Reduced", v) for v in b]
        pvals[label] = stats.mann_whitney(list(a), list(b))
    long = pd.DataFrame(rows, columns=["Metric", "Features", "Score"])
    sns.boxplot(data=long, x="Metric", y="Score", hue="Features", ax=ax,
                palette=["#1b5e63", "#c9c46e"])
    for i, label in enumerate(FIG3A_SCORING):
        ax.text(i, long.Score.max() + 0.008, stats.stars(pvals[label]), ha="center")
    ax.set_ylim(top=long.Score.max() + 0.035)  # headroom so the labels clear the title
    ax.legend(title="Features", loc="lower left")
    ax.set_title("XGBoost: Cross-Validated Performance on Full vs. Reduced Feature Set")
    ax.set_xlabel("")

    # (b) ROC on the hold-out test set
    ax = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, color="#1b5e63", label=f"AUROC = {100 * roc_auc_score(y_test, prob):.1f}%")
    ax.set_xlabel("False Positive Rate (1-Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("AUROC Curve")
    ax.legend(loc="lower right")

    # (c) sensitivity / specificity across thresholds
    ax = axes[1, 0]
    ax.plot(test_curve.threshold, test_curve.sensitivity, color="#1b5e63", label="Sensitivity (Sen)")
    ax.plot(test_curve.threshold, test_curve.specificity, color="#c9c46e", label="Specificity (Spe)")
    for t in (0.10, 0.25, 0.50, 0.75, 0.90):
        s, p, _ = sens_spec(y_test, prob, t)
        ax.axvline(t, color="grey", lw=0.8)
        if abs(t - t_opt) < 0.05:
            continue  # the chosen-threshold label below would overprint this one
        ax.text(t + 0.005, 0.05, f"Sen = {100 * s:.1f}%\nSpe = {100 * p:.1f}%", rotation=90, fontsize=7)
    ax.axvline(t_opt, color="green", lw=1.2)
    ax.text(t_opt + 0.005, 0.05, f"Highest MCC\nSen = {100 * m_opt['sensitivity']:.1f}%\n"
            f"Spe = {100 * m_opt['specificity']:.1f}%", rotation=90, fontsize=7, color="green")
    ax.set_xlabel("Probability of CVD")
    ax.set_ylabel("Score")
    ax.set_title("Sensitivity and Specificity Across Different Probability Thresholds")
    ax.legend(loc="upper left")

    # (d) confusion matrix at the chosen threshold
    ax = axes[1, 1]
    cm = m_opt["confusion_matrix"]
    mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    sns.heatmap(mat, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax,
                xticklabels=["No HD", "HD"], yticklabels=["No HD", "HD"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(f"Confusion Matrix: on Reduced Feature Set, P(CVD) = {100 * t_opt:.0f}%")

    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "fig3_final_model.png", dpi=200)
    plt.close(fig)
    return pvals


def main() -> None:
    X_tr, y_tr = data.load_split("train")
    X_te, y_te = data.load_split("test")
    feats = C.PAPER_TOP5
    tuned_all = json.loads((C.MODELS_DIR / "tuned_params.json").read_text())["params"]["XGBoost"]

    # Tune XGBoost on the five features with the Appendix B grid (10-fold CV, AUROC).
    model, params = tune("XGBoost", X_tr[feats], y_tr)
    print("Final XGBoost hyperparameters:", params)

    # Sec. II-F: threshold chosen on validation data (out-of-fold predictions of
    # the training set) as the one of 100 thresholds maximising MCC.
    oof = cross_val_predict(clone(model), X_tr[feats], y_tr, cv=cv(), method="predict_proba",
                            n_jobs=-1)[:, 1]
    val_curve = curve(y_tr, oof)
    t_opt = float(val_curve.loc[val_curve.mcc.idxmax(), "threshold"])
    print(f"Threshold maximising validation MCC: {t_opt:.4f}  (paper: 0.59)")

    prob = model.predict_proba(X_te[feats])[:, 1]
    test_curve = curve(y_te, prob)
    m_opt = metrics_at(y_te, prob, t_opt)
    m_050 = metrics_at(y_te, prob, 0.5)
    t_test_best = float(test_curve.loc[test_curve.mcc.idxmax(), "threshold"])

    # Fig. 3a: XGBoost full (15) vs reduced (5) features, 10-fold CV on train.
    full_model = clone(model).set_params(**tuned_all)
    full_cv = cross_validate(full_model, X_tr[C.FEATURES], y_tr, cv=cv(), scoring=FIG3A_SCORING,
                             n_jobs=-1)
    red_cv = cross_validate(clone(model), X_tr[feats], y_tr, cv=cv(), scoring=FIG3A_SCORING,
                            n_jobs=-1)
    pvals = fig3(full_cv, red_cv, y_te, prob, test_curve, t_opt, m_opt)

    # Full-feature XGBoost on the test set, for the Sec. III-C comparison
    # (paper: AUROC 94%, precision / recall / F1 88%).
    full_fit = clone(full_model).fit(X_tr[C.FEATURES], y_tr)
    m_full = metrics_at(y_te, full_fit.predict_proba(X_te[C.FEATURES])[:, 1], 0.5)

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, C.MODELS_DIR / "final_xgb.joblib")
    test_curve.to_csv(C.MODELS_DIR / "test_threshold_curve.csv", index=False)
    val_curve.to_csv(C.TABLES_DIR / "validation_threshold_curve.csv", index=False)
    meta = {
        "features": feats, "hyperparameters": params, "threshold": t_opt,
        "threshold_maximising_test_mcc": t_test_best,
        "test_metrics_at_threshold": m_opt, "test_metrics_at_0.5": m_050,
        "full_feature_test_metrics_at_0.5": m_full,
        "fig3a_cv": {k: {"full": list(map(float, full_cv[f"test_{k}"])),
                         "reduced": list(map(float, red_cv[f"test_{k}"])),
                         "mann_whitney_p": pvals[k]} for k in FIG3A_SCORING},
        "scaler": "models/scaler.joblib",
    }
    (C.MODELS_DIR / "final_model_meta.json").write_text(json.dumps(meta, indent=2, default=float))

    def show(title, m):
        print(f"\n--- {title} ---")
        print(f"AUROC {m['AUROC']:.4f} | Sens {m['sensitivity']:.4f} | Spec {m['specificity']:.4f} | "
              f"MCC {m['MCC']:.4f} | FNR {m['false_negative_rate']:.4f} | FPR {m['false_positive_rate']:.4f}")
        print(f"positive class : P {m['precision_positive']:.4f}  R {m['recall_positive']:.4f}  "
              f"F1 {m['f1_positive']:.4f}")
        print(f"weighted avg   : P {m['precision_weighted']:.4f}  R {m['recall_weighted']:.4f}  "
              f"F1 {m['f1_weighted']:.4f}")
        print("confusion matrix:", m["confusion_matrix"])

    show(f"Test set, reduced (5) features, threshold {t_opt:.2f}", m_opt)
    show("Test set, reduced (5) features, threshold 0.50", m_050)
    show("Test set, full (15) features, threshold 0.50", m_full)
    print(f"\nThreshold that would maximise MCC on the test set itself: {t_test_best:.4f} "
          "(reported for comparison only, not used)")
    print("Fig. 3a Mann-Whitney p:", {k: f"{v:.3g}" for k, v in pvals.items()})


if __name__ == "__main__":
    main()
