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
| Top 5 robust features (Table III) | **Yes, same 5 features**; ranks 4 and 5 swapped (R-02) |
| 5 features lose little accuracy vs 15 (Fig. 2, Fig. 7) | **Yes.** Pooled median AUROC 0.909 → 0.895; XGBoost CV curve plateaus after about 5–9 features |
| Final 5-feature XGBoost: test AUROC 91.3% | **Lower on our split: 87.2%.** Within the 95% range of split-to-split variation (85.1–93.5%); the paper's value is also inside it (R-03) |
| SHAP: feature directions and importance order (Fig. 4a) | **Yes, same order and directions** |
| SHAP baseline E[f(x)] = 0.555 | **Yes: 0.553** |
| FHIR-based data exchange + app (Fig. 5) | **Implemented** as a desktop app over a local FHIR R4 server (P-02) |

## Table II: dataset

All 5 numeric means/SDs and all 16 categorical counts match, except two cells
where the paper is wrong: FastingBS=0 / no CVD is 366, not 266; MaxHR / CVD is
127.66 (23.39), not a repeat of the no-CVD value. `reports/tables/table2_*.csv`

## Preprocessing

19 outliers removed (|z| > 3) → 899 patients → 719 train / **180 test**. The
paper's confusion matrix also sums to 180.

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

| Feature | Ours | Paper |
|---|---|---|
| ST_Slope_Flat | 1.19 ± 0.20 | 1.2 ± 0.21 |
| ExerciseAngina_Y | 2.89 ± 0.63 | 2.9 ± 0.36 |
| Sex_M | 4.83 ± 0.52 | 4.3 ± 0.40 |
| ChestPainType_ATA | 4.90 ± 1.17 | 6.7 ± 0.53 |
| Oldpeak | 5.49 ± 0.70 | 5.4 ± 0.78 |
| *6th* | ChestPainType_NAP 7.09 | Cholesterol 6.7 (tied with ATA) |
| AUROC, top 5 (mean over classifiers) | 0.89 | 0.894 |
| AUROC, all features | 0.92 | 0.917 |

`reports/tables/table3_feature_rankings.csv`

## Fig. 2 and Fig. 7: is 5 features enough?

* Fig. 2(a), pooled over all classifiers and methods: median AUROC **0.909 (15
  features) vs 0.895 (top 5)**. Paper: 0.94 vs 0.92.
* As in the paper's Fig. 2(e), the Decision Tree *improves* with the top 5 from
  Logistic L1, Tree-Based, FSFS and BSFS.
* Fig. 7 (XGBoost, 10-fold CV): 1 feature 0.776, 3 → 0.881, **5 → 0.907**, plateau
  ≈ 0.93 from 9–10 features. Paper: ≈0.79, ≈0.89, ≈0.91, plateau ≈0.945.

## Fig. 3: final model (XGBoost, paper's 5 features, 180-patient test set)

| Metric | Ours | Paper |
|---|---|---|
| Test AUROC | **0.872** | 0.913 |
| Threshold (max validation MCC) | 0.49 | 0.59 |
| Sensitivity / specificity | 85.1% / 74.4% | 89.0% / 85.4% |
| FNR / FPR | 14.9% / 25.6% | 11.0% / 14.6% |
| MCC | 0.600 | not reported |
| Precision / recall / F1, weighted avg at 0.5 (A-14) | 80.2% / 80.0% / 79.9% | 86% / 86% / 86% |
| Confusion matrix TN / FP / FN / TP | 64 / 22 / 14 / 80 | 76 / 13 / 10 / 81 |
| Full 15-feature XGBoost test AUROC | 0.925 | 0.94 |

Fig. 3(a), 10-fold CV medians full → reduced: AUROC 0.923 → 0.893, precision
0.884 → 0.849, recall 0.881 → 0.846, F1 0.880 → 0.845. Same direction as the paper,
but not significant (Mann–Whitney p = 0.27–0.65; paper: p < 0.01).

**Split robustness (supplementary, `cvd/step4b_split_robustness.py`).** Repeating
the 80/20 split with 50 seeds (everything else fixed) gives test AUROC
**0.893 ± 0.024** (95% range 0.851–0.935). Our split is at the 18th percentile, the
paper's 0.913 at the 82nd. A single 180-patient test set moves AUROC by several
points on its own, so the gap to the paper is consistent with the choice of split.

## Fig. 4: explanations (KernelSHAP, probability scale)

* Baseline E[f(x)] = **0.553** (paper 0.555).
* Mean |SHAP| order: ST_Slope_Flat 0.176 > ExerciseAngina_Y 0.128 > Oldpeak 0.086
  > Sex_M 0.069 > ChestPainType_ATA 0.045: **same order as the paper's Fig. 4(a)**.
* Directions as in the paper: flat slope, exercise angina, male sex and higher
  Oldpeak raise risk; atypical angina lowers it.
* Example patients: low risk 16.9% (paper 16.9%), high risk 93.9% (paper 93.7%).
* Limitation (R-06): in rare feature combinations, explanations rest on few
  training patients.

## FHIR and the desktop application

* The 180 test patients are stored as FHIR R4 resources (180 `Patient` + 1,620
  `Observation`); read-back through the FHIR API reproduces every CSV value
  (0 mismatches).
* The desktop app loads a patient over FHIR, predicts risk and shows the SHAP
  waterfall, a Fig. 4-style contribution table and the sensitivity/specificity
  curve. For patient cvd-460 it shows 93.9%, identical to the pipeline's Fig. 4(c).
