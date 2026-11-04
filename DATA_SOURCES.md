# Data Sources — Credit Risk Scorecard

## Data actually used
- **UCI Default of Credit Card Clients** (Yeh & Lien, 2009)
  - Source: UCI Machine Learning Repository
  - URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls
  - Rows: 30,000. Target: `default payment next month` (renamed to
    `default_next_month` in data/credit.csv).
  - File in this repo: `data/credit.csv` (2.8 MB).
  - Reproduce it: `python src/download_data.py` (downloads the .xls,
    renames the target, writes data/credit.csv).

## Data NOT used, and why
- **Home Credit Default Risk (300K+ applications)** — the dataset
  named in the original project brief — was NOT used.
  - Reason: Home Credit is distributed via Kaggle, which requires
    Kaggle authentication. No Kaggle credentials were available in
    the build environment, so the dataset could not be downloaded
    without auth, and no Home Credit file, schema, or figure is
    claimed in this repo.
  - Substitution: UCI credit-default data, same task (predict
    next-month default from credit history), smaller scale.
  - Consequence: every metric in outputs/metrics.json and README.md
    is computed on the UCI data (30,000 rows, test set 6,000 rows),
    NOT on Home Credit, and no Home Credit figure is reproduced.

## No other data
No employer, customer, or personal data is used. The UCI dataset
contains no names, addresses, or account numbers; `ID` is a row
identifier and is dropped before modelling.
