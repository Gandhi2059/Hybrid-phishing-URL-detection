"""
scripts/train_svm.py
--------------------
Train a calibrated LinearSVC (SVM) on the full features dataset.

The scaler and model are saved together in a single pickle dict so the
API can reload them atomically.

Usage
-----
    python scripts/train_svm.py
    python scripts/train_svm.py --data data/features.csv --out models/svm_model.pkl
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ── CLI (only parsed when run directly) ───────────────────────────── #
def _build_parser():
    p = argparse.ArgumentParser(description="Train SVM (LinearSVC) on features CSV.")
    p.add_argument("--data", default="data/features.csv")
    p.add_argument("--out",  default="models/svm_model.pkl")
    p.add_argument("--max-iter",    type=int, default=2000)
    p.add_argument("--random-state", type=int, default=42)
    return p


def train_svm(data_path: str, model_path: str, max_iter: int = 2000, random_state: int = 42) -> None:
    print("Training SVM (LinearSVC) on ENTIRE dataset …")
    df = pd.read_csv(data_path)
    X  = df.drop(["url", "label"], axis=1)
    y  = df["label"]
    print(f"  Samples: {len(y):,}  |  Phishing: {y.sum():,}  |  Legit: {(y==0).sum():,}")

    scaler     = StandardScaler()
    X_scaled   = scaler.fit_transform(X)
    base_model = LinearSVC(loss="squared_hinge", dual=False, max_iter=max_iter, random_state=random_state,
                           class_weight="balanced")  # corrects 3.4:1 legit:phishing imbalance
    model      = CalibratedClassifierCV(base_model, cv=3)
    model.fit(X_scaled, y)

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"✔ SVM saved → {model_path}")


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train_svm(args.data, args.out, args.max_iter, args.random_state)
