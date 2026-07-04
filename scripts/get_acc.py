"""
scripts/get_acc.py
------------------
Quick accuracy sanity-check: evaluates all three trained models on a
random 2 000-row sample from data/features.csv.

Usage
-----
    python scripts/get_acc.py
    python scripts/get_acc.py --n-samples 5000
"""

import argparse
import os
import pickle
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import pandas as pd
from src.evaluation import evaluate_model

_p = argparse.ArgumentParser()
_p.add_argument("--data",      default="data/features.csv")
_p.add_argument("--n-samples", type=int, default=2000)
args = _p.parse_args()

ROOT       = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT, "models")

try:
    df = pd.read_csv(os.path.join(ROOT, args.data))
    df = df.sample(n=min(args.n_samples, len(df)), random_state=42)
    X  = df.drop(["url", "label"], axis=1)
    y  = df["label"]

    MODELS = [
        ("SVM",           os.path.join(MODELS_DIR, "svm_model.pkl")),
        ("Random Forest", os.path.join(MODELS_DIR, "rf_model.pkl")),
        ("XGBoost",       os.path.join(MODELS_DIR, "xgb_model.pkl")),
    ]

    print(f"\nEvaluating on {len(y):,} samples …\n")
    for name, path in MODELS:
        if not os.path.exists(path):
            print(f"  [{name}] model not found at {path}")
            continue
        with open(path, "rb") as f:
            payload = pickle.load(f)

        if name == "SVM":
            if isinstance(payload, dict):
                model   = payload["model"]
                scaler  = payload["scaler"]
                features_to_use = getattr(scaler, "feature_names_in_", getattr(model, "feature_names_in_", None))
                if features_to_use is not None:
                    X_eval = X[list(features_to_use)]
                else:
                    X_eval = X
                X_eval  = scaler.transform(X_eval)
            else:
                model  = payload
                features_to_use = getattr(model, "feature_names_in_", None)
                if features_to_use is not None:
                    X_eval = X[list(features_to_use)]
                else:
                    X_eval = X
        else:
            model  = payload
            features_to_use = getattr(model, "feature_names_in_", None)
            if features_to_use is not None:
                X_eval = X[list(features_to_use)]
            else:
                X_eval = X

        preds = model.predict(X_eval)
        res   = evaluate_model(y, preds, name)
        print(f"  {res['Model']:20s}  Accuracy={res['Accuracy']*100:.2f}%  "
              f"F1={res['F1-Score']*100:.2f}%")

except Exception as exc:
    print(f"Error: {exc}")
    sys.exit(1)
