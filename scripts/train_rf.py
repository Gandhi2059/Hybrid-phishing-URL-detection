"""
scripts/train_rf.py
--------------------
Train a Random Forest classifier on the full features dataset.

Usage
-----
    python scripts/train_rf.py
    python scripts/train_rf.py --data data/features.csv --out models/rf_model.pkl
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ── CLI (only parsed when run directly) ─────────────────────────────────── #
def _build_parser():
    p = argparse.ArgumentParser(description="Train Random Forest on features CSV.")
    p.add_argument("--data", default="data/features.csv", help="Path to features CSV")
    p.add_argument("--out",  default="models/rf_model.pkl", help="Output model path")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--random-state", type=int, default=42)
    return p


def train_rf(data_path: str, model_path: str, n_estimators: int = 200, random_state: int = 42) -> None:
    print("Training Random Forest on ENTIRE dataset …")
    df = pd.read_csv(data_path)
    X  = df.drop(["url", "label"], axis=1)
    y  = df["label"]
    print(f"  Samples: {len(y):,}  |  Phishing: {y.sum():,}  |  Legit: {(y==0).sum():,}")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",   # corrects 3.4:1 legit:phishing imbalance
    )
    model.fit(X, y)

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✔ Random Forest saved → {model_path}")


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train_rf(args.data, args.out, args.n_estimators, args.random_state)
