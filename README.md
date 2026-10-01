# Explainable CVD Risk Screening with Five Clinical Features

Final-year project reproducing

> S. Vyshnya, R. Epperson, F. Giuste, W. Shi, A. Hornback, M. D. Wang,
> "Optimized Clinical Feature Analysis for Improved Cardiovascular Disease Risk
> Screening," *IEEE Open Journal of Engineering in Medicine and Biology*, vol. 5,
> pp. 816–827, 2024. DOI: 10.1109/OJEMB.2023.3347479

The project has three parts:

1. **ML pipeline** (`cvd/`): the paper's methodology step by step: preprocessing,
   8 classifiers with nested cross-validation, 7 feature-selection methods with
   rank averaging, the final 5-feature XGBoost model, MCC-optimal threshold and
   KernelSHAP explanations. Every table and figure of the paper is regenerated.
2. **FHIR layer**: the hold-out test patients converted to FHIR R4 `Patient` and
   `Observation` resources and served by a local FHIR REST server, which stands
   in for a hospital EHR.
3. **Desktop application** (`desktop_app/`): a PySide6 app that loads a patient
   from the FHIR server (or takes manual input), predicts CVD risk and explains it
   with SHAP. This is the desktop counterpart of the paper's web app (Fig. 5).

Every assumption, deliberate change and paper erratum is recorded in
[DEVIATIONS.md](DEVIATIONS.md). A side-by-side comparison of the paper's numbers
with ours is in [RESULTS.md](RESULTS.md).

## Setup

Requires Python 3.12 (Windows commands shown).

```
py -V:3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

The dataset `data/raw/heart.csv` is the Kaggle "Heart Failure Prediction" dataset
(fedesoriano, 2021), the paper's ref. [21].

## Running the pipeline

Run from the project root, in order:

| Step | Command | Reproduces | Approx. time (16 cores) |
|------|---------|-----------|------------------------|
| 1 | `.venv\Scripts\python -m cvd.step1_preprocess` | Sec. II-B, Table II | seconds |
| 2 | `.venv\Scripts\python -m cvd.step2_nested_cv` | Sec. II-C, Appendix A/B, Fig. 6 | ~30 min |
| 3 | `.venv\Scripts\python -m cvd.step3_feature_selection` | Sec. II-D, III-A/B, Appendix C/D, Table III, Fig. 2, Fig. 7 | 1–2 h (resumable) |
| 4 | `.venv\Scripts\python -m cvd.step4_final_model` | Sec. II-E/F, III-C, Fig. 3 | ~5 min |
| 4b | `.venv\Scripts\python -m cvd.step4b_split_robustness` | Supplementary: test AUROC over 50 random splits | ~2 min |
| 5 | `.venv\Scripts\python -m cvd.step5_explain` | Sec. III-C-2, Fig. 4 | ~3 min |
| 6 | `.venv\Scripts\python -m cvd.fhir_server` (leave running), then `.venv\Scripts\python -m cvd.step6_fhir` | Sec. IV-B | seconds |

Outputs:

* `reports/tables/`: CSV/JSON tables (Table II, Table III, nested-CV scores, metrics)
* `reports/figures/`: Fig. 2, 3, 4, 6, 7 equivalents
* `models/`: scaler, tuned hyperparameters, feature rankings, final model, threshold curve
* `fhir/`: the FHIR transaction bundle and the server's data store

## Running the desktop app

Double-click `run_app.bat`, or:

```
.venv\Scripts\python -m desktop_app.app
```

1. Click **Start local server** (or run `python -m cvd.fhir_server` yourself).
2. Click **Load test patients** the first time, to upload the 180 hold-out
   patients as FHIR resources.
3. Double-click a patient. Their five features are read from FHIR and the risk,
   SHAP contributions and sensitivity/specificity curve are shown. You can also
   edit any value, or enter a new patient by hand, and press **Assess CVD risk**.

The app is a research prototype and must not be used for clinical decisions.

## Project layout

```
cvd/
  config.py              all settings and assumptions
  data.py                preprocessing (Sec. II-B)
  models.py              8 classifiers + Appendix B grids
  stats.py               Kruskal-Wallis / Dunn / Mann-Whitney
  feature_selection.py   7 selection methods (Appendix C)
  predict.py             risk + KernelSHAP for one patient (shared by app and Fig. 4)
  fhir_mapping.py        heart.csv row <-> FHIR R4 resources
  fhir_server.py         local FHIR R4 REST server
  fhir_client.py         FHIR REST client
  step1..step6_*.py      pipeline steps
desktop_app/app.py       PySide6 desktop application
DEVIATIONS.md            assumptions, decisions, errata
```
