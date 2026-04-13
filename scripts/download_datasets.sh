#!/usr/bin/env bash
set -euo pipefail

mkdir -p datasets

echo "[1/5] Downloading COMPAS (ProPublica compas-scores-two-years.csv)..."
curl -L "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv" \
  -o datasets/compas.csv

echo "[2/5] Downloading Adult dataset from OpenML -> datasets/adult.csv..."
python - <<'PY'
from sklearn.datasets import fetch_openml
import pandas as pd

X, y = fetch_openml(data_id=1119, as_frame=True, return_X_y=True)  # Adult
df = X.copy()
df["income"] = y.astype(str)
df.to_csv("datasets/adult.csv", index=False)
print("Saved datasets/adult.csv", df.shape)
PY

echo "[3/5] Downloading German Credit (OpenML credit-g, data_id=31) -> datasets/german_credit.csv..."
python - <<'PY'
from sklearn.datasets import fetch_openml
import pandas as pd

X, y = fetch_openml(data_id=31, as_frame=True, return_X_y=True)  # credit-g
df = X.copy()
df["target"] = y.astype(str)  # good/bad
df.to_csv("datasets/german_credit.csv", index=False)
print("Saved datasets/german_credit.csv", df.shape)
PY

echo "[4/5] Downloading CrowS-Pairs anonymized CSV..."
curl -L "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv" \
  -o datasets/crows_pairs.csv

echo "[5/5] Downloading BBQ (Bias Benchmark for QA) via HuggingFace datasets..."
python - <<'PY'
import pandas as pd
from datasets import load_dataset

# The BBQ dataset is hosted under multiple repo IDs depending on mirrors.
# We'll try a small list of known IDs and pick the first that works.
CANDIDATES = [
    "Elfsong/BBQ",          # common mirror
    "bbq-lite",             # fallback if someone created a lite copy
    "nyu-mll/bbq",          # sometimes used naming
    "bbq",                  # last-resort (often fails)
]

ds = None
last_err = None
for name in CANDIDATES:
    try:
        ds = load_dataset(name)
        print(f"Loaded BBQ dataset from: {name}")
        break
    except Exception as e:
        last_err = e

if ds is None:
    print("WARNING: Could not download BBQ from HuggingFace.")
    print("Last error:", repr(last_err))
    print("Continuing without BBQ. (Your code can still run; BBQ will be missing.)")
else:
    # Save one split to CSV for reproducibility.
    # Many BBQ repos provide 'train'/'test' or multiple configs; handle both.
    split_name = "test" if "test" in ds else ("train" if "train" in ds else list(ds.keys())[0])
    df = pd.DataFrame(ds[split_name])
    df.to_csv("datasets/bbq.csv", index=False)
    print("Saved datasets/bbq.csv", df.shape)
PY

echo "Done. Files in datasets/:"
ls -lh datasets