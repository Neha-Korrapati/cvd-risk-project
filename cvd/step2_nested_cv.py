"""Step 2: baseline comparison of the 8 classifiers on the full 15-feature set
with nested cross-validation (Sec. II-C, Appendix A, Fig. 6), then tuning of
each classifier on the whole training set (Sec. II-D step 1).

Run from the project root:
    .venv\\Scripts\\python -m cvd.step2_nested_cv
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config as C
from . import data, models, stats


def cv(n_splits: int = 10) -> StratifiedKFold:
    # Appendix A: 10-fold CV in both loops. Fig. 6 caption: "stratified shuffle
    # splits" -> stratified, shuffled K-fold (DEVIATIONS.md A-08).
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=C.SEED)


def tune(name: str, X, y, inner=None):
    """Fit `name` on (X, y), grid-searching its Appendix B grid by AUROC."""
    est, grid = models.make(name)
    if not grid:
        return clone(est).fit(X, y), {}
    gs = GridSearchCV(est, grid, scoring="roc_auc", cv=inner or cv(), n_jobs=-1,
                      error_score=np.nan, refit=True)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


def nested_cv(name: str, X: pd.DataFrame, y: pd.Series) -> dict:
    outer_auc, chosen = [], []
    for tr, va in cv().split(X, y):
        model, params = tune(name, X.iloc[tr], y.iloc[tr])
        outer_auc.append(roc_auc_score(y.iloc[va], model.predict_proba(X.iloc[va])[:, 1]))
        chosen.append(params)
    return {"auroc": outer_auc, "params_per_fold": chosen}


def plot_fig6(scores: dict[str, list[float]], sig: pd.DataFrame) -> None:
    long = pd.DataFrame([(n, v) for n in scores for v in scores[n]], columns=["Classifier", "AUROC"])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=long, x="Classifier", y="AUROC", hue="Classifier", legend=False, ax=ax,
                palette="viridis", showfliers=False)
    sns.stripplot(data=long, x="Classifier", y="AUROC", ax=ax, color="black", size=3)
    top = long["AUROC"].max()
    for i, n in enumerate(scores):
        row = sig[sig["group"] == n]
        if len(row) and row["sig"].iloc[0] != "ns":
            ax.text(i, top + 0.01, row["sig"].iloc[0], ha="center")
    ax.set_ylim(top=top + 0.04)  # headroom so significance stars clear the title
    ax.set_title("Classifier Nested Cross-Validation (full feature set)")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "fig6_nested_cv.png", dpi=200)
    plt.close(fig)


def main() -> None:
    X, y = data.load_split("train")  # the 20% hold-out is never touched here
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in models.CLASSIFIERS:
        t = time.time()
        results[name] = nested_cv(name, X, y)
        a = results[name]["auroc"]
        print(f"{name:20s} AUROC median {np.median(a):.4f}  mean {np.mean(a):.4f} "
              f"± {np.std(a):.4f}   ({time.time() - t:.0f}s)", flush=True)

    scores = {n: r["auroc"] for n, r in results.items()}
    kw_p, sig = stats.kruskal_dunn(scores, reference="Logistic Regression")
    plot_fig6(scores, sig)

    summary = pd.DataFrame({n: {"median": np.median(v), "mean": np.mean(v), "std": np.std(v)}
                            for n, v in scores.items()}).T
    summary.to_csv(C.TABLES_DIR / "nested_cv_auroc.csv")
    sig.to_csv(C.TABLES_DIR / "nested_cv_dunn_vs_lr.csv", index=False)
    (C.TABLES_DIR / "nested_cv_raw.json").write_text(json.dumps(results, indent=2, default=float))

    best = summary["median"].idxmax()
    print(f"\nKruskal-Wallis p = {kw_p:.3g}")
    print(sig.to_string(index=False))
    print(f"\nBest classifier by median nested-CV AUROC: {best}")

    # Sec. II-D step 1: tune every classifier on the whole training set; these
    # tuned models are the ones used by the feature-selection pipeline.
    tuned = {}
    for name in models.CLASSIFIERS:
        _, params = tune(name, X, y)
        tuned[name] = params
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (C.MODELS_DIR / "tuned_params.json").write_text(
        json.dumps({"best_classifier": best, "params": tuned}, indent=2, default=float))
    print("\nTuned hyperparameters (full training set):")
    print(json.dumps(tuned, indent=2, default=float))


if __name__ == "__main__":
    main()
