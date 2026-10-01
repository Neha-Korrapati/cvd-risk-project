"""Single source of truth for every setting and every assumption.

Anything the paper does not state is marked ASSUMPTION and is also listed in
DEVIATIONS.md, so the reader can see exactly where we had to fill a gap.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "heart.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"

# ASSUMPTION: the paper reports no random seed.
SEED = 42

TARGET = "HeartDisease"

# Paper Sec. II-B: 5 numeric and 6 categorical features (Table I).
NUMERIC = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CATEGORICAL = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ST_Slope", "ExerciseAngina"]

# Paper Sec. II-B: "Categorical features were binarized with one of the
# categories dropped to create a reference group". The dropped level is not
# named in the text, but Table III lists exactly which dummies survived, so the
# reference groups below are the ones absent from Table III.
REFERENCE_LEVEL = {
    "Sex": "F",
    "ChestPainType": "ASY",
    "FastingBS": 0,
    "RestingECG": "Normal",
    "ST_Slope": "Up",
    "ExerciseAngina": "N",
}

# The 15 model features, named exactly as in Table III.
FEATURES = [
    "Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak",
    "Sex_M",
    "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
    "FastingBS",
    "RestingECG_LVH", "RestingECG_ST",
    "ST_Slope_Down", "ST_Slope_Flat",
    "ExerciseAngina_Y",
]

# Paper Sec. II-B: numeric features z-scored; any row with |z| > 3 on at least
# one numeric feature is an outlier.
Z_THRESHOLD = 3.0

# Paper Sec. II-B: 80/20 train / hold-out split.
TEST_SIZE = 0.20
# ASSUMPTION: stratification is not stated. The paper's test confusion matrix
# (Fig. 3d) has 91 CVD / 89 no-CVD = 50.6% CVD, while the data is 55% CVD, which
# points to a plain (non-stratified) random split, so we do not stratify.
STRATIFY_SPLIT = False

# Paper Sec. III-C and Table III: the five features of the final model.
# Project decision: the final model uses the paper's five features; our own
# ranking (Table III reproduction) is reported alongside for comparison.
PAPER_TOP5 = ["ST_Slope_Flat", "ExerciseAngina_Y", "Sex_M", "Oldpeak", "ChestPainType_ATA"]
