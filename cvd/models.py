"""Paper Sec. II-C and Appendix B: the eight shallow classifiers and their
hyperparameter settings.

Appendix B is partly garbled in the published PDF. Each reading below is the
literal text interpreted with the paper's own notation ("10E3" = 10^3, as in
"10E-3 to 1" for learning rates), and is logged in DEVIATIONS.md (E-08, A-04..A-07).

Appendix B text (verbatim)                          -> our grid
LR:  penalty = l2, intercept scaling=1, class weight=None -> fixed, not tuned
XGB: learning rate = 20 values from 1E-3 to 1 (log)       -> 20 values logspace(-3, 0)
     n estimators (number of trees) = 10000               -> fixed 10000
KNN: no. nearest neighbors = 15 values from 1 to 10E3 (log) -> 15 ints logspace(0, 3)
     weights = ['uniform', 'distance']                     -> both
SVC: kernel = rbf, gamma = 25 values 10E-10 to 10E5 (log) -> 25 values logspace(-10, 5)
     max iterations = 100                                  -> fixed 100
DT:  gini, best, max_depth None, min_samples_split 2, min_samples_leaf 1 -> fixed
AdaBoost: n estimators = 7 values from 1 to 10E3 (log)    -> 7 ints logspace(0, 3)
     learning rate = 10 values from 10E-3 to 1 (log)       -> 10 values logspace(-3, 0)
RF, Gaussian NB: not listed                                -> library defaults, not tuned
"""
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import SEED


def _int_logspace(start: float, stop: float, num: int) -> list[int]:
    return sorted({int(round(v)) for v in np.logspace(start, stop, num)})


# Display names follow Fig. 6 / Table III.
CLASSIFIERS = ["Logistic Regression", "XGBoost", "KNN", "SVC Radial",
               "Decision Tree", "RF", "AdaBoost", "Gaussian NB"]


def make(name: str):
    """Return (untuned estimator, param_grid). An empty grid means not tuned."""
    if name == "Logistic Regression":
        # max_iter raised only so lbfgs converges; it does not change the model.
        return LogisticRegression(penalty="l2", intercept_scaling=1, class_weight=None,
                                  max_iter=5000, random_state=SEED), {}
    if name == "XGBoost":
        return (XGBClassifier(n_estimators=10000, tree_method="hist", eval_metric="logloss",
                              random_state=SEED, n_jobs=1),
                {"learning_rate": np.logspace(-3, 0, 20).tolist()})
    if name == "KNN":
        # Values of k larger than an inner training fold are invalid; GridSearchCV
        # scores them NaN and they can never be selected.
        return (KNeighborsClassifier(),
                {"n_neighbors": _int_logspace(0, 3, 15), "weights": ["uniform", "distance"]})
    if name == "SVC Radial":
        # probability=True so the model exposes predict_proba (needed for
        # KernelSHAP and for probability thresholds); AUROC is unaffected.
        return (SVC(kernel="rbf", max_iter=100, probability=True, random_state=SEED),
                {"gamma": np.logspace(-10, 5, 25).tolist()})
    if name == "Decision Tree":
        return DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None,
                                      min_samples_split=2, min_samples_leaf=1,
                                      random_state=SEED), {}
    if name == "RF":
        return RandomForestClassifier(random_state=SEED, n_jobs=1), {}
    if name == "AdaBoost":
        return (AdaBoostClassifier(random_state=SEED),
                {"n_estimators": _int_logspace(0, 3, 7),
                 "learning_rate": np.logspace(-3, 0, 10).tolist()})
    if name == "Gaussian NB":
        return GaussianNB(), {}
    raise ValueError(name)
