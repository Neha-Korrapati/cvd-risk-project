"""Step 3: robust feature-selection pipeline (Sec. II-D, III-A, III-B,
Appendix C, Appendix D) producing Table III, Fig. 2 and Fig. 7.

Rankings are cached in models/rankings.json, so an interrupted run resumes
where it stopped.

Run from the project root:
    .venv\\Scripts\\python -m cvd.step3_feature_selection
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib.externals.loky import get_reusable_executor
from joblib.externals.loky.process_executor import BrokenProcessPool
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import config as C
from . import data, models, stats
from . import feature_selection as fs

CACHE = C.MODELS_DIR / "rankings.json"

# Table III as printed (column 9, average rank across classifiers), used to
# draw the paper's ordering in Fig. 7 and for the side-by-side comparison.
PAPER_TABLE3_AVG = {
    "ST_Slope_Flat": 1.2, "ExerciseAngina_Y": 2.9, "Sex_M": 4.3, "Oldpeak": 5.4,
    "ChestPainType_ATA": 6.7, "Cholesterol": 6.7, "FastingBS": 7.3,
    "ChestPainType_NAP": 8.3, "Age": 9.0, "MaxHR": 9.1, "RestingBP": 11.0,
    "ST_Slope_Down": 11.4, "RestingECG_LVH": 11.6, "RestingECG_ST": 11.9,
    "ChestPainType_TA": 12.7,
}


def tuned_model(name: str, params: dict):
    est, _ = models.make(name)
    return clone(est).set_params(**params)


def cv10() -> StratifiedKFold:
    return StratifiedKFold(10, shuffle=True, random_state=C.SEED)


def cv_auc(model, X, y, cols, retries: int = 3) -> list[float]:
    # Retry with a fresh worker pool if a Windows joblib worker dies (see
    # feature_selection._parallel_scores); the result is deterministic.
    for attempt in range(retries):
        try:
            return cross_val_score(clone(model), X[list(cols)], y, scoring="roc_auc", cv=cv10(),
                                   n_jobs=-1).tolist()
        except BrokenProcessPool:
            if attempt == retries - 1:
                raise
            get_reusable_executor().shutdown(wait=True)


def compute_rankings(X, y, tuned: dict) -> dict:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    def save():
        CACHE.write_text(json.dumps(cache, indent=2))

    # F-test, Logistic L1 and Tree-Based do not depend on the classifier, so
    # they give the same ranking for every classifier; compute them once.
    shared = cache.setdefault("_shared", {})
    for method, fn in (("F-Test", fs.ftest_rank), ("Logistic L1", fs.l1_rank),
                       ("Tree-Based", fs.tree_rank)):
        if method not in shared:
            t = time.time()
            shared[method] = fn(X, y).to_dict()
            save()
            print(f"  {method:12s} (shared)          {time.time() - t:6.0f}s", flush=True)

    for name in models.CLASSIFIERS:
        model = tuned_model(name, tuned[name])
        per = cache.setdefault(name, {})
        for method in fs.METHODS:
            if method in per or method in shared:
                continue
            if method == "RFE" and name in fs.NO_RFE:
                continue
            t = time.time()
            if method == "SHAP":
                r = fs.shap_rank(clone(model).fit(X, y), X)
            elif method == "RFE":
                r = fs.rfe_rank(model, X, y)
            elif method == "FSFS":
                r = fs.forward_rank(model, X, y)
            else:
                r = fs.backward_rank(model, X, y)
            per[method] = r.to_dict()
            save()
            print(f"  {method:12s} {name:20s} {time.time() - t:6.0f}s", flush=True)

    out = {}
    for name in models.CLASSIFIERS:
        methods = {m: shared[m] for m in shared}
        methods.update(cache[name])
        out[name] = {m: methods[m] for m in fs.METHODS if m in methods}
    return out


def table3(rankings: dict, X, y, tuned: dict) -> pd.DataFrame:
    """Table III: average rank per classifier over its methods, then the mean
    and standard deviation across classifiers. Decision Tree is excluded from
    the averaging because of its poor performance (Sec. III-B)."""
    included = [n for n in models.CLASSIFIERS if n != "Decision Tree"]
    per_clf = pd.DataFrame({n: pd.DataFrame(rankings[n]).mean(axis=1) for n in included})
    per_clf = per_clf.loc[C.FEATURES]
    tbl = per_clf.copy()
    tbl["Average (all classifiers)"] = per_clf.mean(axis=1)
    # Sample std (ddof=1) reproduces the paper's column 10 (e.g. 0.21 for ST_Slope_Flat).
    tbl["Stdev (all classifiers)"] = per_clf.std(axis=1, ddof=1)
    tbl = tbl.sort_values("Average (all classifiers)")

    auc5, aucall = {}, {}
    for n in included:
        model = tuned_model(n, tuned[n])
        top5 = per_clf[n].sort_values().index[:5]
        auc5[n] = np.mean(cv_auc(model, X, y, top5))
        aucall[n] = np.mean(cv_auc(model, X, y, C.FEATURES))
    for label, vals in (("AUROC (top 5 features)", auc5), ("AUROC (all features)", aucall)):
        row = pd.Series(vals)
        row["Average (all classifiers)"] = row[included].mean()
        row["Stdev (all classifiers)"] = row[included].std(ddof=1)
        tbl.loc[label] = row
    return tbl


def fig2(rankings: dict, X, y, tuned: dict) -> pd.DataFrame:
    """Fig. 2: 10-fold CV AUROC of each classifier on all features ("None") and
    on the top 5 features chosen by each selection method."""
    rows = []
    for name in models.CLASSIFIERS:
        model = tuned_model(name, tuned[name])
        rows += [(name, "None", v) for v in cv_auc(model, X, y, C.FEATURES)]
        for method, r in rankings[name].items():
            top5 = pd.Series(r).sort_values().index[:5]
            rows += [(name, method, v) for v in cv_auc(model, X, y, top5)]
    long = pd.DataFrame(rows, columns=["Classifier", "Method", "AUROC"])
    order = ["None"] + fs.METHODS

    fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharey=True)
    panels = [("All Classifiers", long)] + [(n, long[long.Classifier == n]) for n in models.CLASSIFIERS]
    sig_rows = []
    for ax, (title, df) in zip(axes.flat, panels):
        present = [m for m in order if m in set(df.Method)]
        sns.boxplot(data=df, x="Method", y="AUROC", order=present, hue="Method", hue_order=present,
                    legend=False, ax=ax, palette="viridis", showfliers=False)
        sns.stripplot(data=df, x="Method", y="AUROC", order=present, ax=ax, color="black", size=2)
        groups = {m: df[df.Method == m]["AUROC"].tolist() for m in present}
        kw_p, sig = stats.kruskal_dunn(groups, reference="None")
        for i, m in enumerate(present):
            s = sig[sig.group == m]
            if len(s) and s.sig.iloc[0] != "ns":
                ax.text(i, 1.005, s.sig.iloc[0], ha="center", fontsize=8)
            if len(s):
                sig_rows.append({"panel": title, "method": m, "kruskal_p": kw_p,
                                 "dunn_p_vs_none": s.p_vs_reference.iloc[0], "sig": s.sig.iloc[0]})
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylim(0.6, 1.02)
        plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "fig2_full_vs_reduced.png", dpi=200)
    plt.close(fig)
    long.to_csv(C.TABLES_DIR / "fig2_auroc_long.csv", index=False)
    pd.DataFrame(sig_rows).to_csv(C.TABLES_DIR / "fig2_significance.csv", index=False)
    return long


def fig7(order_ours: list[str], X, y, tuned: dict) -> pd.DataFrame:
    """Fig. 7 / Appendix D: XGBoost AUROC as features are added in Table III order."""
    model = tuned_model("XGBoost", tuned["XGBoost"])
    order_paper = sorted(PAPER_TABLE3_AVG, key=lambda f: (PAPER_TABLE3_AVG[f],
                                                         list(PAPER_TABLE3_AVG).index(f)))
    rows = []
    for k in range(1, len(C.FEATURES) + 1):
        rows.append({"n_features": k,
                     "AUROC (our Table III order)": np.mean(cv_auc(model, X, y, order_ours[:k])),
                     "AUROC (paper Table III order)": np.mean(cv_auc(model, X, y, order_paper[:k]))})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df.n_features, df["AUROC (our Table III order)"], marker="o", label="Our Table III order")
    ax.plot(df.n_features, df["AUROC (paper Table III order)"], marker="s", ls="--",
            label="Paper Table III order")
    ax.axvline(5, color="grey", ls=":")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC vs. Number of Features (XGBoost)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "fig7_auroc_vs_n_features.png", dpi=200)
    plt.close(fig)
    df.to_csv(C.TABLES_DIR / "fig7_auroc_vs_n_features.csv", index=False)
    return df


def main() -> None:
    X, y = data.load_split("train")
    tuned = json.loads((C.MODELS_DIR / "tuned_params.json").read_text())["params"]

    print("Computing feature rankings (7 methods x 8 classifiers)...", flush=True)
    rankings = compute_rankings(X, y, tuned)

    tbl = table3(rankings, X, y, tuned)
    tbl.round(4).to_csv(C.TABLES_DIR / "table3_feature_rankings.csv")
    pd.set_option("display.width", 250)
    print("\n=== Table III (reproduced) ===")
    print(tbl.round(2).to_string())

    feat_rows = tbl.drop(index=["AUROC (top 5 features)", "AUROC (all features)"])
    order_ours = feat_rows.index.tolist()
    comparison = pd.DataFrame({
        "our avg rank": feat_rows["Average (all classifiers)"].round(2),
        "our position": range(1, 16),
        "paper avg rank": pd.Series(PAPER_TABLE3_AVG),
    })
    comparison.to_csv(C.TABLES_DIR / "table3_vs_paper.csv")
    print("\nOur top 5:  ", order_ours[:5])
    print("Paper top 5:", C.PAPER_TOP5)
    print("Overlap:    ", sorted(set(order_ours[:5]) & set(C.PAPER_TOP5)))

    long = fig2(rankings, X, y, tuned)
    full = long[long.Method == "None"].AUROC.median()
    reduced = long[long.Method != "None"].AUROC.median()
    print(f"\nFig. 2(a) pooled median AUROC: full {full:.3f} vs top-5 {reduced:.3f} "
          f"(paper: 0.94 vs 0.92)")

    df7 = fig7(order_ours, X, y, tuned)
    print("\n=== Fig. 7 ===")
    print(df7.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
