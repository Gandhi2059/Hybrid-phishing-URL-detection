"""
scripts/train_xgb.py
--------------------
Train an XGBoost classifier on the full features dataset.

Usage
-----
    python scripts/train_xgb.py
    python scripts/train_xgb.py --data data/features.csv --out models/xgb_model.pkl
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import xgboost as xgb

# ── CLI (only parsed when run directly) ───────────────────────────── #
def _build_parser():
    p = argparse.ArgumentParser(description="Train XGBoost on features CSV.")
    p.add_argument("--data", default="data/features.csv")
    p.add_argument("--out",  default="models/xgb_model.pkl")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--random-state", type=int, default=42)
    return p


def train_xgb(data_path: str, model_path: str, n_estimators: int = 300, random_state: int = 42) -> None:
    print("Training XGBoost on ENTIRE dataset …")
    df = pd.read_csv(data_path)
    X  = df.drop(["url", "label"], axis=1)
    y  = df["label"]
    print(f"  Samples: {len(y):,}  |  Phishing: {y.sum():,}  |  Legit: {(y==0).sum():,}")

    # scale_pos_weight = count(negatives)/count(positives) — XGBoost's equivalent
    # of class_weight='balanced'; corrects the 3.4:1 legit:phishing imbalance.
    neg  = int((y == 0).sum())
    pos  = int((y == 1).sum())
    spw  = round(neg / pos, 4)
    print(f"  scale_pos_weight = {spw}  (neg={neg:,} / pos={pos:,})")

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=random_state,
        verbosity=0,
        scale_pos_weight=spw,   # corrects 3.4:1 legit:phishing imbalance
    )
    model.fit(X, y)

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✔ XGBoost saved → {model_path}")


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train_xgb(args.data, args.out, args.n_estimators, args.random_state)
