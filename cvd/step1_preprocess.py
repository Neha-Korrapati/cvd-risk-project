"""Step 1: preprocessing (Sec. II-B) and reproduction of Table II.

Run from the project root:
    .venv\\Scripts\\python -m cvd.step1_preprocess
"""
import json

import pandas as pd

from . import config as C
from . import data

# Table II exactly as printed in the paper: (no-CVD, CVD).
PAPER_NUMERIC = {
    "Age": ((50.55, 9.44), (55.90, 8.73)),
    "RestingBP": ((130.18, 16.50), (134.19, 19.83)),
    "Cholesterol": ((227.12, 74.63), (175.94, 126.39)),
    "MaxHR": ((148.15, 23.29), (148.15, 23.29)),
    "Oldpeak": ((0.41, 0.70), (1.27, 1.15)),
}
PAPER_COUNTS = {
    ("Sex", "F"): (143, 50), ("Sex", "M"): (267, 458),
    ("ChestPainType", "TA"): (26, 20), ("ChestPainType", "ATA"): (149, 24),
    ("ChestPainType", "NAP"): (131, 72), ("ChestPainType", "ASY"): (104, 392),
    ("FastingBS", 0): (266, 338), ("FastingBS", 1): (44, 170),
    ("RestingECG", "Normal"): (267, 285), ("RestingECG", "ST"): (61, 117),
    ("RestingECG", "LVH"): (82, 106),
    ("ST_Slope", "Up"): (317, 78), ("ST_Slope", "Flat"): (79, 381),
    ("ST_Slope", "Down"): (14, 49),
    ("ExerciseAngina", "N"): (355, 192), ("ExerciseAngina", "Y"): (55, 316),
}


def table2(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(raw)
    groups = {0: raw[raw[C.TARGET] == 0], 1: raw[raw[C.TARGET] == 1]}

    num_rows = []
    for feat, paper in PAPER_NUMERIC.items():
        row = {"Feature": feat}
        for cls, label in ((0, "No CVD"), (1, "CVD")):
            m, s = groups[cls][feat].mean(), groups[cls][feat].std()
            pm, ps = paper[cls]
            row[f"{label} (ours)"] = f"{m:.2f} ({s:.2f})"
            row[f"{label} (paper)"] = f"{pm:.2f} ({ps:.2f})"
            row[f"{label} match"] = round(m, 2) == pm and round(s, 2) == ps
        num_rows.append(row)

    cat_rows = []
    for (feat, lvl), paper in PAPER_COUNTS.items():
        row = {"Feature": feat, "Level": lvl}
        for cls, label in ((0, "No CVD"), (1, "CVD")):
            k = int((groups[cls][feat] == lvl).sum())
            row[f"{label} (ours)"] = f"{k} ({100 * k / n:.0f}%)"
            row[f"{label} (paper)"] = f"{paper[cls]} ({100 * paper[cls] / n:.0f}%)"
            row[f"{label} match"] = k == paper[cls]
        cat_rows.append(row)

    return pd.DataFrame(num_rows), pd.DataFrame(cat_rows)


def main() -> None:
    raw = data.load_raw()
    summary = data.run()

    num, cat = table2(raw)
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    num.to_csv(C.TABLES_DIR / "table2_numeric.csv", index=False)
    cat.to_csv(C.TABLES_DIR / "table2_categorical.csv", index=False)
    (C.TABLES_DIR / "preprocess_summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    print("=== Preprocessing summary ===")
    print(json.dumps(summary, indent=2))
    print("\n=== Table II: numeric features, mean (std) ===")
    print(num.to_string(index=False))
    print("\n=== Table II: categorical features, # patients (% of all 918) ===")
    print(cat.to_string(index=False))
    mismatches = pd.concat([
        num.loc[~(num["No CVD match"] & num["CVD match"]), ["Feature"]],
        cat.loc[~(cat["No CVD match"] & cat["CVD match"]), ["Feature", "Level"]],
    ])
    print(f"\nCells differing from the printed paper table: {len(mismatches)}")
    print(mismatches.to_string(index=False))


if __name__ == "__main__":
    main()
