"""
scripts/train_all.py
---------------------
Full-dataset training + 5-fold stratified cross-validated evaluation.

Trains Random Forest, XGBoost, SVM (LinearSVC), and optionally the
custom Logistic Regression on 100 % of data/features.csv, then reports
Accuracy, Precision, Recall, F1, and AUC via cross-validation so there
is no information leak from a held-out test split.

Usage
-----
    python scripts/train_all.py
    python scripts/train_all.py --rebuild-features
    python scripts/train_all.py --cv-only
    python scripts/train_all.py --no-cv
    python scripts/train_all.py --cv-folds 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import xgboost as xgb

from src.preprocessing import clean_data
from src.feature_engineering import FeatureExtractor
from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_metrics,
    save_metrics_to_csv,
)
from scripts.train_rf  import train_rf
from scripts.train_svm import train_svm
from scripts.train_xgb import train_xgb

# ── Paths ────────────────────────────────────────────────────────────────── #
ROOT        = os.path.join(os.path.dirname(__file__), "..")
URLS_CSV    = os.path.join(ROOT, "data", "urls.csv")
FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")
MODELS_DIR  = os.path.join(ROOT, "models")
GRAPHS_DIR  = os.path.join(ROOT, "graphs")

RF_PATH    = os.path.join(MODELS_DIR, "rf_model.pkl")
XGB_PATH   = os.path.join(MODELS_DIR, "xgb_model.pkl")
SVM_PATH   = os.path.join(MODELS_DIR, "svm_model.pkl")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")


# ── Helpers ──────────────────────────────────────────────────────────────── #
def banner(msg: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {msg}")
    print("=" * 65)


def build_features(rebuild: bool = False) -> pd.DataFrame:
    if os.path.exists(FEATURES_CSV) and not rebuild:
        print(f"  ✔ Loading pre-built features from {FEATURES_CSV}")
        df = pd.read_csv(FEATURES_CSV)
        print(f"    → {len(df):,} rows  |  {df.shape[1]} columns")
        return df

    if not os.path.exists(URLS_CSV):
        sys.exit(
            f"ERROR: Neither {FEATURES_CSV} nor {URLS_CSV} found.\n"
            "Run:  python scripts/prep_data.py"
        )

    print(f"  Building features from {URLS_CSV} …")
    df = pd.read_csv(URLS_CSV)
    df = clean_data(df)
    print(f"  Dataset after cleaning: {len(df):,} rows")

    extractor = FeatureExtractor(use_tfidf=True, max_tfidf_features=100)
    extractor.fit_tfidf(df["url"].tolist())
    df_feat = extractor.extract_features(df)

    os.makedirs(MODELS_DIR, exist_ok=True)
    extractor.save_vectorizer(TFIDF_PATH)
    df_feat.to_csv(FEATURES_CSV, index=False)
    print(f"  ✔ Saved {len(df_feat):,} rows → {FEATURES_CSV}")
    print(f"  ✔ TF-IDF vectorizer  → {TFIDF_PATH}")
    return df_feat


def run_cross_validation(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> list[dict]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    results: list[dict] = []

    # Random Forest
    print(f"\n  → {n_splits}-fold CV on Random Forest …")
    t0 = time.time()
    rf_preds = cross_val_predict(
        RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42,
                               class_weight="balanced"),  # handles 3.4:1 imbalance
        X, y, cv=cv, n_jobs=-1,
    )
    print(f"     done in {time.time()-t0:.1f}s")
    results.append(evaluate_model(y, rf_preds, "Random Forest (5-fold CV)"))
    plot_confusion_matrix(y, rf_preds, "Random Forest", GRAPHS_DIR)

    # XGBoost — scale_pos_weight = neg/pos (XGBoost's class_weight='balanced')
    _neg, _pos = int((y == 0).sum()), int((y == 1).sum())
    _spw = round(_neg / _pos, 4)
    print(f"\n  → {n_splits}-fold CV on XGBoost (scale_pos_weight={_spw}) …")
    t0 = time.time()
    xgb_preds = cross_val_predict(
        xgb.XGBClassifier(n_estimators=300, eval_metric="logloss",
                          n_jobs=-1, random_state=42, verbosity=0,
                          scale_pos_weight=_spw),  # handles 3.4:1 imbalance
        X, y, cv=cv, n_jobs=-1,
    )
    print(f"     done in {time.time()-t0:.1f}s")
    results.append(evaluate_model(y, xgb_preds, "XGBoost (5-fold CV)"))
    plot_confusion_matrix(y, xgb_preds, "XGBoost", GRAPHS_DIR)

    # SVM
    print(f"\n  → {n_splits}-fold CV on SVM (LinearSVC) …")
    t0 = time.time()
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            LinearSVC(dual=False, max_iter=2000, random_state=42,
                      class_weight="balanced"),  # handles 3.4:1 imbalance
            cv=3
        )),
    ])
    svm_preds = cross_val_predict(svm_pipe, X, y, cv=cv, n_jobs=-1)
    print(f"     done in {time.time()-t0:.1f}s")
    results.append(evaluate_model(y, svm_preds, "SVM Linear (5-fold CV)"))
    plot_confusion_matrix(y, svm_preds, "SVM", GRAPHS_DIR)

    # Logistic Regression (from scratch)
    try:
        from src.logistic_scratch import LogisticRegressionScratch
        print(f"\n  → {n_splits}-fold CV on Logistic Regression (scratch) …")
        t0     = time.time()
        X_np   = X.to_numpy()
        y_np   = y.to_numpy()
        lr_pred = np.zeros(len(y_np), dtype=int)
        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X_np)
        for tr_idx, te_idx in cv.split(X_sc, y_np):
            lr = LogisticRegressionScratch(learning_rate=0.1, num_iterations=200)
            lr.fit(X_sc[tr_idx], y_np[tr_idx])
            lr_pred[te_idx] = lr.predict(X_sc[te_idx])
        print(f"     done in {time.time()-t0:.1f}s")
        results.append(evaluate_model(y_np, lr_pred, "Logistic Reg. Scratch (5-fold CV)"))
        plot_confusion_matrix(y_np, lr_pred, "Logistic Reg. (Scratch)", GRAPHS_DIR)
    except Exception as exc:
        print(f"  [Skip] LR Scratch: {exc}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────── #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train all models on the full dataset.")
    parser.add_argument("--rebuild-features", action="store_true",
                        help="Force re-extraction of features from data/urls.csv")
    parser.add_argument("--cv-only", action="store_true",
                        help="Skip training; run cross-validation only")
    parser.add_argument("--no-cv",   action="store_true",
                        help="Skip cross-validation; just train models")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    args = parser.parse_args()

    total_start  = time.time()
    TOTAL_STEPS  = 4 if not (args.cv_only or args.no_cv) else 3

    banner("Phishing Detection  —  Full Dataset Training  (train_all.py)")

    # Step 1: Features
    print(f"\n[1/{TOTAL_STEPS}] Loading / building features …")
    df_feat = build_features(rebuild=args.rebuild_features)
    X = df_feat.drop(["url", "label"], axis=1)
    y = df_feat["label"]
    print(f"\n  Dataset summary:")
    print(f"    Rows     : {len(y):,}")
    print(f"    Features : {X.shape[1]}")
    print(f"    Phishing : {y.sum():,}  ({y.mean()*100:.1f}%)")
    print(f"    Legit    : {(y==0).sum():,}  ({(y==0).mean()*100:.1f}%)")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    # Step 2: Train on 100 % of data
    if not args.cv_only:
        print(f"\n[2/{TOTAL_STEPS}] Training all models on 100 % of the dataset …")
        t0 = time.time()
        train_rf (FEATURES_CSV, RF_PATH)
        train_xgb(FEATURES_CSV, XGB_PATH)
        train_svm(FEATURES_CSV, SVM_PATH)
        print(f"\n  ✔ All models trained in {time.time()-t0:.1f}s")

    # Step 3: Cross-validated evaluation
    if not args.no_cv:
        print(f"\n[{3 if not args.cv_only else 2}/{TOTAL_STEPS}] "
              f"Running {args.cv_folds}-fold stratified CV …")
        results    = run_cross_validation(X, y, n_splits=args.cv_folds)
        results_df = pd.DataFrame(results)
        plot_metrics(results_df, GRAPHS_DIR)
        save_metrics_to_csv(results, GRAPHS_DIR, "metrics_summary.csv")

        print("\n" + "─" * 65)
        print(f"{'Model':<35} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        print("─" * 65)
        for r in results:
            print(f"  {r['Model']:<33} {r['Accuracy']:>7.4f} {r['Precision']:>7.4f}"
                  f" {r['Recall']:>7.4f} {r['F1-Score']:>7.4f}")
        print("─" * 65)

    elapsed = time.time() - total_start
    banner(f"Pipeline complete  ({elapsed:.1f}s total)")
    print(f"  Trained models → {MODELS_DIR}/")
    print(f"  Graphs & CSVs  → {GRAPHS_DIR}/")
    print()


if __name__ == "__main__":
    main()
