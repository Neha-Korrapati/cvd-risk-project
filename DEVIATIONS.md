# Deviations, assumptions and paper errata

Reproduction of Vyshnya et al., *Optimized Clinical Feature Analysis for Improved
Cardiovascular Disease Risk Screening*, IEEE OJEMB 5 (2024) 816–827,
DOI 10.1109/OJEMB.2023.3347479.

Every place where this project departs from, or has to fill a gap in, the paper is
listed here. Settings live in `cvd/config.py`.

## Dataset

Kaggle "Heart Failure Prediction" (fedesoriano, 2021), `heart.csv`, the paper's
ref. [21]. 918 rows, 11 features, no missing values.
SHA-256: `948420B084D8A3A0CA42B8419FCE9AEE175879E43F8AEDF712377899A67AA49B`.

## Assumptions (gaps in the paper)

| ID | Topic | Paper says | What we do | Why |
|----|-------|-----------|------------|-----|
| A-01 | Random seed | Not reported | `SEED = 42` everywhere | Needed for repeatability |
| A-02 | Reference categories | "one of the categories was dropped" | F, ASY, FastingBS=0, Normal, Up, N | These are exactly the dummies missing from Table III |
| A-03 | Split stratification | Not stated | Plain random split, not stratified | Paper's test set is 91/89 (50.6% CVD) vs 55% overall, which suggests no stratification |
| A-04 | Appendix B grids | Garbled text (E-08) | Literal reading with the paper's own "10E3 = 10^3" notation; see the table in `cvd/models.py` | Closest reading of the printed text |
| A-05 | XGBoost trees | "n estimators = 10000" | Fixed 10,000 trees, learning rate tuned over 20 log-spaced values in [1e-3, 1] | Literal reading. Costs about 30 min of nested CV on 16 cores |
| A-06 | RF and Gaussian NB | No hyperparameters given | Library defaults, not tuned | Nothing to reproduce |
| A-07 | KNN grid | k from 1 to 10^3 | k values larger than an inner training fold (611, 1000) fail and are scored NaN, so they can never be selected | Literal grid kept; invalid points dropped automatically |
| A-08 | CV splitter | Appendix A: 10-fold; Fig. 6: "stratified shuffle splits" | Stratified, shuffled 10-fold in both nested loops | Satisfies both statements |
| A-09 | Tuning metric / search | Not stated | Exhaustive grid search, scored by AUROC | AUROC is the paper's primary metric |
| A-10 | Logistic L1 selector | C not given | C tuned over 20 values by 10-fold CV (AUROC); rank by \|coefficient\| | |
| A-11 | Tree-Based selector | Importance type not given | XGBoost default importance ("gain") | |
| A-12 | SHAP selector | "average SHAP values"; background not given | Mean \|SHAP\|; KernelSHAP with a 20-centre k-means background, explaining 100 training rows | Mean \|SHAP\| is the standard global importance. Sizes keep KernelSHAP on 15 features tractable |
| A-13 | FSFS / BSFS | Scoring AUROC; CV folds not given | 10-fold stratified CV per candidate subset (matching the 10-fold CV the paper uses everywhere else; an earlier run used 5-fold, kept in `models/rankings_sfs5fold.json`). BSFS rank = reverse removal order | Greedy selection is deterministic, so one full pass gives exactly the ranks of the paper's n = 1..15 procedure |
| A-14 | Precision / recall / F1 | Test values "86%" for all three | Reported both positive-class and support-weighted. The paper's 86/86/86 only fits weighted averages at the 0.5 threshold | Derived from the paper's own confusion matrix |
| A-15 | SHAP for the final model | Background not given | All training patients as background, so E[f(x)] is the mean predicted risk ("average risk across all patients", Fig. 4). With 5 features KernelSHAP enumerates all 32 coalitions, so it is exact | |
| A-16 | FHIR resources and codes | Not given | FHIR R4 `Patient` + one `Observation` per feature. LOINC 8480-6 (systolic BP) and 2093-3 (total cholesterol); a local CodeSystem for exercise-test / ECG findings | See `cvd/fhir_mapping.py` |
| A-17 | Where selection runs | Not stated | Nested CV, tuning and all feature selection use only the 80% training set; the 20% test set is used once, in Step 4 | Avoids test leakage beyond D-02 |
| A-18 | Rank ties | Not stated | Tied scores share the average rank (e.g. several L1 coefficients shrunk to 0) | |

## Deliberate project decisions

| ID | Decision | Paper | Ours |
|----|----------|-------|------|
| P-01 | Final model features | Top 5 from its own ranking | The paper's 5 (`ST_Slope_Flat, ExerciseAngina_Y, Sex_M, Oldpeak, ChestPainType_ATA`), chosen by the project owner. Our own ranking is still computed and reported for comparison. |
| P-02 | Application | Web app (SMART-on-FHIR) | Desktop app (PySide6) acting as a FHIR REST client against a local FHIR-compatible server. No SMART OAuth. |
| P-03 | Oldpeak input | Fig. 5 accepts the standardized value | App accepts Oldpeak in mV (as in the dataset) and standardizes it internally with the saved scaler |

## Faithful-but-flawed steps (reproduced as written, flagged)

| ID | Issue |
|----|-------|
| D-02 | Scaling and outlier detection are fit on all 918 rows before the train/test split, so test-set statistics leak into the scaler. Reproduced as the paper describes. |

## Results that differ from the paper

| ID | Item | Paper | Ours | How we proceed |
|----|------|-------|------|----------------|
| R-01 | Best classifier in nested CV (Fig. 6) | XGBoost clearly highest (≈0.94) | XGBoost has the highest **mean** AUROC (0.926), but by **median** it is 0.0005 behind Logistic Regression and AdaBoost (0.923). Only Decision Tree differs significantly from LR (Dunn p = 0.001), as in the paper | The final model uses XGBoost, as the paper does, and the mean-based ranking supports it. We report that our data supports this choice only narrowly |
| R-02 | Top-5 features (Table III) | ST_Slope_Flat, ExerciseAngina_Y, Sex_M, Oldpeak, ChestPainType_ATA | **Same five features.** Ranks 1–2 match (ST_Slope_Flat 1.19 vs 1.2; ExerciseAngina_Y 2.83 vs 2.9); ranks 3–5 are reshuffled (ATA 4.45, Sex_M 4.95, Oldpeak 5.71). With 5-fold SFS (A-13, earlier run) only ranks 4 and 5 were swapped, showing these three are close and sensitive to implementation details. The paper's ATA/Cholesterol tie at 6.7 does not occur (Cholesterol 8th, 8.17); 6th place (ChestPainType_NAP 6.86) is 1.15 ranks below 5th | Our own pipeline independently supports the project decision P-01 |
| R-03 | Hold-out test AUROC of the final model | 0.913 | **0.872** (sens 85.1% / spec 74.4% at threshold 0.49) | Split robustness (`cvd/step4b_split_robustness.py`): over 50 random 80/20 splits the same model gives 0.893 ± 0.024 (95% range 0.851–0.935). Our split is at the 18th percentile and the paper's value at the 82nd, so the gap is consistent with split-to-split variation on a 180-patient test set. The 5-feature 10-fold CV AUROC is 0.907 (Fig. 7). We report the primary split as-is and did not tune on it |
| R-04 | MCC-optimal threshold | 0.59 | 0.49 (chosen on out-of-fold validation predictions, Sec. II-F) | Threshold depends on the fitted model and the data split; the paper also notes it "may vary based on the specific clinical context" |
| R-05 | Fig. 3a full vs reduced (10-fold CV) | Significant drops with 5 features (** / ***) | Same direction (median AUROC 0.923 → 0.893) but not significant (Mann–Whitney p = 0.27–0.65) | Reported as found |
| R-06 | SHAP for sparse subgroups | Paper notes effects vary per patient | For the low-risk example (atypical angina, no exercise angina, flat slope), Oldpeak = 1.8 mV **lowers** risk by 23 points. Only 17 training patients share this profile (2 with Oldpeak > 1.5), so this is an interaction learned from very little data | Stated as a limitation: explanations in rare feature combinations rest on few patients |

## Errata found in the paper

| ID | Location | Printed | Correct (verified against data) |
|----|----------|---------|---------------------------------|
| E-01 | Sec. II-B text | 45% have CVD, 55% do not | 508/918 = **55% have CVD** (also matches SHAP baseline 0.555) |
| E-02 | Sec. II-B text | "a total of 16 features" | **15** (Table III, ranks 1–15) |
| E-03 | Table I | RestingBP = diastolic BP | Resting **systolic** BP in the UCI source |
| E-04 | Table II | FastingBS=0, no CVD: 266 | **366** |
| E-05 | Table II | MaxHR, CVD: 148.15 (23.29) | **127.66 (23.39)** (the no-CVD value was repeated) |
| E-06 | Sec. IV-A | Sensitivity/specificity 91.21% / 80.90% | Results section and Fig. 3 report **89.01% / 85.39%** at the chosen 59% threshold; 91/81 is the 50% threshold |
| E-07 | Sec. II-B text | "Duplicate observations and missing data were removed" | The Kaggle file's source (UCI, 920 rows) has only 531 complete rows; missing values were filled by the Kaggle author, not removed. Not an issue for this project since we use the Kaggle file directly. |
| E-08 | Appendix B | e.g. "learning rate = 20", "n estimators = 10000", "penalty = 12" | Garbled; interpretation documented with the hyperparameter grids (Step 2) |
