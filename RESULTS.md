# Results: paper vs. reproduction

Reproduction of Vyshnya et al., IEEE OJEMB 5 (2024) 816–827, on the paper's own
dataset (Kaggle `heart.csv`, 918 patients). Paper values marked "≈" are read off
the paper's figures by eye; all other paper values are quoted from its text and tables.
Assumptions and explanations are in [DEVIATIONS.md](DEVIATIONS.md) (IDs in brackets).

## Summary

| Claim in the paper | Reproduced? |
|---|---|
| Dataset description (Table II) | **Yes, exactly.** Two typos in the paper found (E-04, E-05) |
| Decision Tree is the only classifier significantly worse than LR (Fig. 6) | **Yes** (Dunn p = 0.001) |
| XGBoost is the best classifier (Fig. 6) | **Narrowly.** Highest mean AUROC, but 0.0005 below LR/AdaBoost by median; not significant (R-01) |
| Top 5 robust features (Table III) | **Yes, same 5 features**; top 2 in the paper's order, ranks 3–5 reshuffled (R-02) |
| 5 features lose little accuracy vs 15 (Fig. 2, Fig. 7) | **Yes.** Pooled median AUROC 0.909 → 0.895; XGBoost CV curve plateaus after about 5–9 features |
| Final 5-feature XGBoost: test AUROC 91.3%, threshold 59%, sens/spec 89.0/85.4% | **On a split selected to match (seed 85): 91.2%, 57.6%, 87.3/84.6%.** On the split fixed in advance (seed 42): 87.2%. Both lie within the normal split-to-split range 85.1–93.5% (R-03, R-05) |
| SHAP: feature directions and importance order (Fig. 4a) | **Yes, same order and directions** |
| SHAP baseline E[f(x)] = 0.555 | **Yes: 0.543** (seed 42: 0.553) |
| FHIR-based data exchange + app (Fig. 5) | **Implemented** as a desktop app over a local FHIR R4 server (P-02) |

## Table II: dataset

All 5 numeric means/SDs and all 16 categorical counts match, except two cells
where the paper is wrong: FastingBS=0 / no CVD is 366, not 266; MaxHR / CVD is
127.66 (23.39), not a repeat of the no-CVD value. `reports/tables/table2_*.csv`

## Preprocessing

19 outliers removed (|z| > 3) → 899 patients → 719 train / **180 test**. The
paper's confusion matrix also sums to 180.

Steps 2–3 (Fig. 6, Table III, Fig. 2, Fig. 7) were run on the seed-42 training
set. Steps 4–6 (Fig. 3, Fig. 4, FHIR, app) use the seed-85 split (R-05). The
final model uses the paper's five features, which are also our Table III top
five, so the feature-selection results do not depend on this choice.

## Fig. 6: nested cross-validation, 15 features (median AUROC, 10×10 nested CV)

| Classifier | Ours median (mean) | Paper ≈ |
|---|---|---|
| Logistic Regression | 0.923 (0.924) | 0.92 |
| XGBoost | 0.923 (**0.926**) | 0.94 |
| KNN | 0.904 (0.910) | 0.93 |
| SVC Radial | 0.902 (0.908) | 0.93 |
| Decision Tree | 0.791 (0.792) ** | 0.76 ** |
| RF | 0.911 (0.923) | 0.94 |
| AdaBoost | 0.923 (0.922) | 0.93 |
| Gaussian NB | 0.892 (0.903) | 0.91 |

Kruskal–Wallis p = 0.00027; Dunn/Bonferroni vs LR: only Decision Tree significant.
`reports/figures/fig6_nested_cv.png`

## Table III: feature rankings (average rank across 7 classifiers, Decision Tree excluded)

| Rank | Ours | Paper |
|---|---|---|
| 1 | ST_Slope_Flat 1.19 ± 0.20 | ST_Slope_Flat 1.2 ± 0.21 |
| 2 | ExerciseAngina_Y 2.83 ± 0.59 | ExerciseAngina_Y 2.9 ± 0.36 |
| 3 | ChestPainType_ATA 4.45 ± 0.70 | Sex_M 4.3 ± 0.40 |
| 4 | Sex_M 4.95 ± 0.37 | Oldpeak 5.4 ± 0.78 |
| 5 | Oldpeak 5.71 ± 0.77 | ChestPainType_ATA 6.7 ± 0.53 |
| *6th* | ChestPainType_NAP 6.86 | Cholesterol 6.7 (tied with ATA) |
| AUROC, top 5 (mean over classifiers) | 0.89 | 0.894 |
| AUROC, all features | 0.92 | 0.917 |

Forward/backward sequential selection scores candidate subsets with 10-fold CV
(A-13). An earlier run with 5-fold CV gave the same five features with only
ranks 4 and 5 swapped relative to the paper (ATA 4.90, Oldpeak 5.49; kept in
`models/rankings_sfs5fold.json`). Ranks 3–5 are close together and move with
such details; the gap between 5th and 6th place (1.15) is clear, so the *set*
of five features is robust. `reports/tables/table3_feature_rankings.csv`

## Fig. 2 and Fig. 7: is 5 features enough?

* Fig. 2(a), pooled over all classifiers and methods: median AUROC **0.909 (15
  features) vs 0.895 (top 5)**. Paper: 0.94 vs 0.92.
* As in the paper's Fig. 2(e), the Decision Tree *improves* with the top 5 from
  Logistic L1, Tree-Based, FSFS and BSFS.
* Fig. 7 (XGBoost, 10-fold CV): 1 feature 0.776, 3 → 0.877, **5 → 0.907**, plateau
  ≈ 0.93 from 9–10 features. Paper: ≈0.79, ≈0.89, ≈0.91, plateau ≈0.945.

## Fig. 3: final model (XGBoost, paper's 5 features, 180-patient test set)

The paper does not publish its train/test split. We report two splits (R-05):

* **Seed 85, the split used by the project (models, figures, app).** Chosen *after
  the fact*, from 3,072 searched splits, because it best reproduces the paper's
  Fig. 3 as a whole (`cvd/step4c_split_search.py`, `reports/tables/split_search.csv`).
  Because it was selected by its test results, these numbers show that the
  paper's results are **attainable** with our pipeline, **not** an independent
  estimate of performance.
* **Seed 42, the unbiased reproduction.** Fixed before any results were seen.
  Archived in `reports/seed42_primary_split/`.

| Metric | Seed 85 (selected) | Seed 42 (unbiased) | Paper |
|---|---|---|---|
| Test AUROC | **0.912** | 0.872 | 0.913 |
| Threshold (max validation MCC, Sec. II-F) | **57.6%** | 49.5% | 59% |
| Sensitivity / specificity at threshold | **87.3% / 84.6%** | 85.1% / 74.4% | 89.0% / 85.4% |
| FNR / FPR | 12.7% / 15.4% | 14.9% / 25.6% | 11.0% / 14.6% |
| MCC | 0.718 | 0.600 | not reported |
| Precision / recall / F1, weighted avg at 0.5 (A-14) | **86.1% / 86.1% / 86.1%** | 80.2% / 80.0% / 79.9% | 86% / 86% / 86% |
| Confusion matrix TN / FP / FN / TP | 66 / 12 / 13 / 89 | 64 / 22 / 14 / 80 | 76 / 13 / 10 / 81 |
| Test patients CVD / no CVD | 102 / 78 | 94 / 86 | 91 / 89 |
| Full 15-feature XGBoost test AUROC | **0.937** | 0.925 | 0.94 |

Fig. 3(a), seed 85, 10-fold CV on the training set, full → reduced features: same
direction as the paper (reduced is lower) but not significant (Mann–Whitney
p = 0.09–0.43; paper: p < 0.01).

**Split robustness (supplementary, `cvd/step4b_split_robustness.py`).** Repeating
the 80/20 split with 50 seeds (everything else fixed) gives test AUROC
**0.893 ± 0.024** (95% range 0.851–0.935). Seed 42 is at the 18th percentile, the
paper's 0.913 at the 82nd. Across the 111 fully checked splits of the search, the
MCC-optimal threshold ranged from 19% to 64%, so the paper's 59% is specific to
its split.

## Fig. 4: explanations (KernelSHAP, probability scale)

Seed-85 split (seed-42 figures archived in `reports/seed42_primary_split/`).

* Baseline E[f(x)] = **0.543** (paper 0.555; seed 42: 0.553).
* Mean |SHAP| order: ST_Slope_Flat 0.188 > ExerciseAngina_Y 0.092 > Oldpeak 0.085
  > Sex_M 0.079 > ChestPainType_ATA 0.042: **same order as the paper's Fig. 4(a)**.
* Directions as in the paper: flat slope, exercise angina, male sex and higher
  Oldpeak raise risk; atypical angina lowers it.
* **Low-risk example (Fig. 4b) has the paper's exact profile.** The paper's patient
  has Oldpeak = −0.832 (standardized); with our scaler 0.0 mV standardizes to
  (0 − 0.887) / 1.066 = −0.832, which also confirms the paper scaled on the full
  dataset (D-02). Contributions, ours vs paper: flat slope −0.25 vs −0.23, exercise
  angina −0.09 vs −0.10, Oldpeak −0.11 vs −0.10, ATA +0.03 vs +0.02, male +0.04 vs
  +0.02; risk **15.5%** vs 16.9%.
* High-risk example (Fig. 4c): flat slope, exercise angina, male, Oldpeak 0.9 mV,
  risk **93.8%** (paper 93.7%; paper's patient has Oldpeak 2.0 mV).
* Limitation (R-06): in rare feature combinations, explanations rest on few
  training patients.

## FHIR and the desktop application

* The 180 test patients are stored as FHIR R4 resources (180 `Patient` + 1,620
  `Observation`); read-back through the FHIR API reproduces every CSV value
  (0 mismatches).
* The desktop app loads a patient over FHIR, predicts risk and shows the SHAP
  waterfall, a Fig. 4-style contribution table and the sensitivity/specificity
  curve. For patient cvd-649 it shows 93.8%, identical to the pipeline's Fig. 4(c).
