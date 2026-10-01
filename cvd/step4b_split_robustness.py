"""Supplementary analysis (not in the paper): how much does the hold-out test
AUROC of the final model depend on which 20% of patients land in the test set?

The paper reports a single split and publishes no seed, so our split cannot
match it. Here we repeat the paper's 80/20 split with 50 seeds, keeping every
other setting fixed (same preprocessing, the paper's five features, the
hyperparameters chosen in Step 4), and look at the spread of test AUROC.
This does NOT replace the primary Step 4 result; it only puts it in context.

Run from the project root:
    .venv\\Scripts\\python -m cvd.step4b_split_robustness
"""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from . import config as C
from . import data, models

N_SEEDS = 50


def one_split(seed: int, kept: pd.DataFrame, params: dict) -> dict:
    tr, te = train_test_split(kept, test_size=C.TEST_SIZE, random_state=seed)
    est, _ = models.make("XGBoost")
    m = clone(est).set_params(**params).fit(tr[C.PAPER_TOP5], tr[C.TARGET])
    return {"seed": seed,
            "test_auroc": roc_auc_score(te[C.TARGET], m.predict_proba(te[C.PAPER_TOP5])[:, 1]),
            "test_cvd_fraction": te[C.TARGET].mean()}


def main() -> None:
    raw = data.load_raw()
    scaled = data.scale(data.encode(raw), data.fit_scaler(data.encode(raw)))
    kept = scaled.loc[~data.outlier_mask(scaled)]
    params = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())["hyperparameters"]

    rows = Parallel(n_jobs=-1)(delayed(one_split)(s, kept, params) for s in range(N_SEEDS))
    df = pd.DataFrame(rows)
    df.to_csv(C.TABLES_DIR / "split_robustness.csv", index=False)

    ours = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    ours_auc = ours["test_metrics_at_threshold"]["AUROC"]
    a = df.test_auroc
    print(f"Test AUROC over {N_SEEDS} random 80/20 splits (paper's 5 features, XGBoost {params}):")
    print(f"  mean {a.mean():.4f}  sd {a.std():.4f}  median {a.median():.4f}  "
          f"min {a.min():.4f}  max {a.max():.4f}")
    print(f"  2.5-97.5 percentile: {np.percentile(a, 2.5):.4f} - {np.percentile(a, 97.5):.4f}")
    print(f"  our primary split (seed {C.SEED}): {ours_auc:.4f} -> percentile "
          f"{100 * (a < ours_auc).mean():.0f}")
    print(f"  paper's reported value 0.913 -> percentile {100 * (a < 0.913).mean():.0f}")


if __name__ == "__main__":
    main()
