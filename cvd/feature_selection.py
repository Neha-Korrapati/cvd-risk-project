"""Paper Sec. II-D and Appendix C: the seven feature-selection methods.

Every function returns a pandas Series indexed by feature name holding the rank
(1 = most important, 15 = least important), computed on the training set only.
"""
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from joblib.externals.loky.process_executor import BrokenProcessPool
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.feature_selection import RFE, f_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     StratifiedShuffleSplit, cross_val_score)

from . import config as C
from . import models

METHODS = ["SHAP", "F-Test", "Logistic L1", "Tree-Based", "RFE", "FSFS", "BSFS"]

# Appendix C-5: RFE needs model weights; KNN, SVC and Gaussian NB have none.
NO_RFE = {"KNN", "SVC Radial", "Gaussian NB"}

# ASSUMPTIONS for details Appendix C does not give (DEVIATIONS.md A-09..A-13).
SHAP_BACKGROUND_K = 20      # k-means summary of the training set
SHAP_EXPLAIN_N = 100        # training rows explained per classifier
SFS_CV_FOLDS = 5            # inner CV used to score candidate subsets


def _rank_desc(scores: pd.Series) -> pd.Series:
    """Rank scores so the largest is 1; ties share their average rank."""
    return pd.Series(rankdata(-scores.values, method="average"), index=scores.index)


def shap_rank(model, X: pd.DataFrame) -> pd.Series:
    """Appendix C-1: KernelSHAP, features ranked by mean |SHAP value|."""
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=-1)  # faster prediction only; the fitted model is unchanged
    background = shap.kmeans(X, SHAP_BACKGROUND_K)
    explain = X.sample(n=min(SHAP_EXPLAIN_N, len(X)), random_state=C.SEED)
    f = lambda a: model.predict_proba(pd.DataFrame(a, columns=X.columns))[:, 1]
    explainer = shap.KernelExplainer(f, background)
    values = explainer.shap_values(explain, silent=True)
    return _rank_desc(pd.Series(np.abs(values).mean(axis=0), index=X.columns))


def ftest_rank(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Appendix C-2: univariate F-test, larger F = more important."""
    f, _ = f_classif(X, y)
    return _rank_desc(pd.Series(f, index=X.columns))


def l1_rank(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Appendix C-3: L1-regularised logistic regression, ranked by |weight|.
    ASSUMPTION A-10: C chosen by 10-fold CV on AUROC."""
    lr = LogisticRegressionCV(Cs=20, penalty="l1", solver="liblinear", scoring="roc_auc",
                              cv=StratifiedKFold(10, shuffle=True, random_state=C.SEED),
                              random_state=C.SEED, max_iter=5000).fit(X, y)
    return _rank_desc(pd.Series(np.abs(lr.coef_[0]), index=X.columns))


def tree_rank(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Appendix C-4: XGBoost feature importances. Hyperparameters tuned with a
    stratified shuffle split holding out 10% (as stated), using the Appendix B
    XGBoost grid. ASSUMPTION A-11: default importance type ("gain")."""
    est, grid = models.make("XGBoost")
    gs = GridSearchCV(est, grid, scoring="roc_auc", n_jobs=-1,
                      cv=StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=C.SEED))
    gs.fit(X, y)
    return _rank_desc(pd.Series(gs.best_estimator_.feature_importances_, index=X.columns))


def rfe_rank(model, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Appendix C-5: the paper runs RFE for n = 1..15 and records which feature
    is added at each n. RFE's elimination path is deterministic, so this is
    exactly sklearn's ranking_ from a single run down to one feature."""
    rfe = RFE(clone(model), n_features_to_select=1, step=1).fit(X, y)
    return pd.Series(rfe.ranking_.astype(float), index=X.columns)


def _subset_auc(model, X, y, cols, cv) -> float:
    return cross_val_score(clone(model), X[list(cols)], y, scoring="roc_auc", cv=cv).mean()


def _parallel_scores(model, X, y, subsets, cv, retries: int = 3) -> list[float]:
    """Score candidate subsets in parallel. On Windows a joblib worker can die
    after a long idle period (BrokenProcessPool, "handle is invalid"); the
    computation is deterministic, so we simply retry with a fresh pool."""
    for attempt in range(retries):
        try:
            return Parallel(n_jobs=-1)(delayed(_subset_auc)(model, X, y, s, cv) for s in subsets)
        except BrokenProcessPool:
            if attempt == retries - 1:
                raise
            get_reusable_executor().shutdown(wait=True)


def forward_rank(model, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Appendix C-6: greedy forward selection scored by AUROC. The order in
    which features enter is their rank (the n = 1..15 procedure of C-5)."""
    cv = StratifiedKFold(SFS_CV_FOLDS, shuffle=True, random_state=C.SEED)
    chosen, remaining = [], list(X.columns)
    while remaining:
        scores = _parallel_scores(model, X, y, [chosen + [f] for f in remaining], cv)
        best = remaining[int(np.argmax(scores))]
        chosen.append(best)
        remaining.remove(best)
    return pd.Series({f: i + 1.0 for i, f in enumerate(chosen)})[X.columns]


def backward_rank(model, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Appendix C-7: greedy backward elimination scored by AUROC; the feature
    whose removal hurts least goes first. Removed first = rank 15, last feature
    standing = rank 1 (ranked by the same n = 1..15 strategy as RFE)."""
    cv = StratifiedKFold(SFS_CV_FOLDS, shuffle=True, random_state=C.SEED)
    kept, removed = list(X.columns), []
    while len(kept) > 1:
        scores = _parallel_scores(model, X, y, [[c for c in kept if c != f] for f in kept], cv)
        drop = kept[int(np.argmax(scores))]
        removed.append(drop)
        kept.remove(drop)
    order = kept + removed[::-1]  # most important first
    return pd.Series({f: i + 1.0 for i, f in enumerate(order)})[X.columns]
