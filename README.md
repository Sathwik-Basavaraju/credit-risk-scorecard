# Explainable Credit Risk Scorecard with Cost-Optimized Approval Policy

Python, XGBoost, scikit-learn, SHAP — built from scratch, locally.

## What this is
A credit-default scorecard that predicts next-month default, explains
declines with SHAP reason codes, and sets an approval cutoff from a
documented cost matrix — not from a 0.5 default threshold.

## Data actually used — read this first
**NOT Home Credit.** The original brief named Home Credit Default Risk
(300K+ applications), but Home Credit is Kaggle-gated and no Kaggle
credentials were available, so it was NOT downloaded and is NOT claimed
here. See DATA_SOURCES.md for the full substitution note.

Data actually used:
- **UCI Default of Credit Card Clients** — 30,000 rows, 22.12% default
  in the full file, 22.12% in the test set (see metrics.json).
- File: `data/credit.csv` (2.8 MB). Reproduce it with
  `python src/download_data.py`.
- Features (23): LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0–PAY_6,
  BILL_AMT1–6, PAY_AMT1–6. Target: `default_next_month`.

## Results — all computed by src/train.py on 2026-11-04 (see outputs/metrics.json)
- Split: stratified 80/20, seed 42. Train: 24,000. **Test: 6,000.**
- Logistic regression AUC: **0.708**
- Primary model (XGBoost, 300 trees, depth 4) AUC: **0.778**
- Calibrated XGBoost AUC: 0.778
- Brier score, primary model: **0.1351** (calibrated version: 0.1359,
  slightly worse — the uncalibrated model is reported as primary,
  calibration curve is in outputs/charts/calibration.png).
- Cost model (documented assumptions, not bank policy):
  LGD = 0.45, foregone margin = 0.12, exposure = LIMIT_BAL.
  Cost-optimal threshold = 0.12 / (0.45 + 0.12) = **0.2105**.
- At that threshold: approval rate **68.55%** (4,113 approved,
  1,887 declined out of 6,000 test applicants).
- **Credit loss at constant approval rate: NT$37,224,000 for the
  cost-optimal policy vs NT$44,634,456 for a PAY_0 single-feature
  baseline approving the same 68.55% — a 16.60% reduction.**
  Credit loss = sum over approved actual defaulters of 0.45 × LIMIT_BAL.
  Foregone interest (declined good applicants, illustrative) was
  NT$14,305,200 for the cost-optimal policy vs NT$12,508,800 for the
  baseline — the cost-optimal policy trades higher foregone interest
  for lower credit loss, as the cost matrix intends.
- Adverse-action reason codes: SHAP (shap.TreeExplainer on the XGBoost
  model) was computed for **all 1,887 declined test applicants** in this
  run. `outputs/reason_codes_sample.csv` contains the first **200**
  of those declined applicants, each with its top 3 SHAP reasons
  (10.6% of declined applicants in the sample file; 100% had SHAP
  values computed in the run). No SHAP values were faked.

## How to run
```bash
pip install -r requirements.txt
python src/download_data.py   # optional; data/credit.csv is already included
python src/train.py           # writes outputs/metrics.json + charts + reason codes
```
Note on XGBoost install: the default wheel pulls a large CUDA
dependency. A CPU-only install (`pip install --no-deps xgboost`)
works for this project. If XGBoost cannot be imported, train.py
falls back to HistGradientBoostingClassifier and records that in
metrics.json — in this run XGBoost was used.

## Files
- `src/download_data.py` — UCI download + conversion
- `src/cost_policy.py` — LGD/margin/threshold and credit-loss maths
- `src/train.py` — training, calibration, cost policy, SHAP reason codes
- `data/credit.csv` — UCI data (30,000 rows)
- `outputs/metrics.json` — all numbers above, with method and test size
- `outputs/charts/roc.png`, `calibration.png`, `shap_importance.png`
- `outputs/reason_codes_sample.csv` — 200 declined applicants, top-3 SHAP reasons each
- `DATA_SOURCES.md` — Home Credit substitution note
- `MANIFEST.txt` — files to upload to GitHub
