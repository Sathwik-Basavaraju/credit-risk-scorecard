"""Download the credit dataset used by this project.

Source actually used (documented alternative, NOT Home Credit):
  UCI Machine Learning Repository — Default of Credit Card Clients
  (Yeh & Lien, 2009), 30,000 rows, downloaded as .xls from:
  https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls

Why not Home Credit Default Risk (300K+ applications)?
  Home Credit is distributed via Kaggle, which requires Kaggle
  authentication. No Kaggle credentials were available in this
  environment, so Home Credit was NOT downloaded and is NOT claimed
  here. The UCI dataset is a real, public credit-default dataset
  with the same task (predict next-month default from credit
  history), at a smaller scale (30,000 rows, not 300,000+).

Usage:
    python src/download_data.py
    # writes data/credit.csv with the target renamed to
    # default_next_month, and prints shape / default rate.
"""

import urllib.request
from pathlib import Path

import pandas as pd

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "credit.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / "data" / "_credit.xls"
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, tmp)
    df = pd.read_excel(tmp, header=1, engine="xlrd")
    df = df.rename(columns={"default payment next month": "default_next_month"})
    df.to_csv(OUT, index=False)
    tmp.unlink(missing_ok=True)
    print(f"Wrote {OUT} shape={df.shape}")
    print(f"Default rate: {df['default_next_month'].mean():.4f}")


if __name__ == "__main__":
    main()
