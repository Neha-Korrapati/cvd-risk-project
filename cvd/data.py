"""Paper Sec. II-B: data description and processing.

Order of operations follows the paper's text:
    encode categoricals -> standard-scale numerics -> drop |z| > 3 rows -> 80/20 split
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config as C


def load_raw() -> pd.DataFrame:
    return pd.read_csv(C.RAW_CSV)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals, dropping the reference level of each."""
    out = df[C.NUMERIC].copy()
    for col in C.CATEGORICAL:
        ref = C.REFERENCE_LEVEL[col]
        levels = sorted(df[col].unique(), key=str)
        if set(levels) <= {0, 1}:  # FastingBS is already binary 0/1
            out[col] = df[col].astype(int)
            continue
        for lvl in levels:
            if lvl != ref:
                out[f"{col}_{lvl}"] = (df[col] == lvl).astype(int)
    out[C.TARGET] = df[C.TARGET].astype(int)
    return out[C.FEATURES + [C.TARGET]]


def fit_scaler(encoded: pd.DataFrame) -> StandardScaler:
    return StandardScaler().fit(encoded[C.NUMERIC])


def scale(encoded: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    out = encoded.copy()
    out[C.NUMERIC] = scaler.transform(encoded[C.NUMERIC])
    return out


def outlier_mask(scaled: pd.DataFrame) -> pd.Series:
    """True for rows with |z| > threshold on at least one numeric feature."""
    return (scaled[C.NUMERIC].abs() > C.Z_THRESHOLD).any(axis=1)


def run() -> dict:
    """Build and save the processed train/test sets. Returns a summary dict."""
    raw = load_raw()
    encoded = encode(raw)

    # The paper scales the full dataset before outlier detection and before the
    # split (DEVIATIONS.md, D-02: this leaks test-set statistics into scaling).
    scaler = fit_scaler(encoded)
    scaled = scale(encoded, scaler)

    is_outlier = outlier_mask(scaled)
    kept = scaled.loc[~is_outlier]

    stratify = kept[C.TARGET] if C.STRATIFY_SPLIT else None
    train, test = train_test_split(kept, test_size=C.TEST_SIZE,
                                   random_state=C.SPLIT_SEED, stratify=stratify)

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Keep the original row index so every processed row can be traced back to
    # heart.csv (needed later to build FHIR patients from raw test-set values).
    train.to_csv(C.PROCESSED_DIR / "train.csv", index_label="row_id")
    test.to_csv(C.PROCESSED_DIR / "test.csv", index_label="row_id")
    raw.loc[is_outlier].to_csv(C.PROCESSED_DIR / "outliers_removed.csv", index_label="row_id")
    joblib.dump(scaler, C.MODELS_DIR / "scaler.joblib")

    per_feature = (scaled[C.NUMERIC].abs() > C.Z_THRESHOLD).sum().to_dict()
    return {
        "raw_rows": len(raw),
        "outliers_removed": int(is_outlier.sum()),
        "outliers_by_feature": {k: int(v) for k, v in per_feature.items()},
        "rows_after_outliers": len(kept),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_cvd": int(train[C.TARGET].sum()),
        "test_cvd": int(test[C.TARGET].sum()),
        "test_no_cvd": int((test[C.TARGET] == 0).sum()),
        "n_features": len(C.FEATURES),
        "scaler_mean": dict(zip(C.NUMERIC, np.round(scaler.mean_, 4).tolist())),
        "scaler_std": dict(zip(C.NUMERIC, np.round(scaler.scale_, 4).tolist())),
    }


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load processed 'train' or 'test' as (X, y) with the 15 paper features."""
    df = pd.read_csv(C.PROCESSED_DIR / f"{name}.csv", index_col="row_id")
    return df[C.FEATURES], df[C.TARGET]
