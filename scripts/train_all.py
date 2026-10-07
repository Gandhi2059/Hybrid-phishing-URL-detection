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

def run_cross_validation(X: pd.DataFrame, y: pd.Series, urls: pd.Series, n_splits: int = 5) -> list[dict]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    results: list[dict] = []

    # Drop globally-fitted TF-IDF features to prevent leakage; we compute them strictly per-fold
    tfidf_cols = [c for c in X.columns if c.startswith("tfidf_")]
    X_base = X.drop(columns=tfidf_cols)
    
    rf_preds  = np.zeros(len(y), dtype=int)
    xgb_preds = np.zeros(len(y), dtype=int)
    svm_preds = np.zeros(len(y), dtype=int)
    lr_preds  = np.zeros(len(y), dtype=int)

    from sklearn.feature_extraction.text import TfidfVectorizer
    
    print(f"\n  → Running {n_splits}-fold CV with STRICT fold-isolated TF-IDF and Scaling …")
    t0 = time.time()
    
    y_np = y.to_numpy()
    urls_np = urls.to_numpy()
    
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X_base, y_np), 1):
        print(f"    Fold {fold}/{n_splits}...")
        X_tr_base, X_te_base = X_base.iloc[tr_idx], X_base.iloc[te_idx]
        y_tr = y_np[tr_idx]
        
        # 1. Strict Fold-Isolated TF-IDF
        tfidf = TfidfVectorizer(max_features=100, analyzer="char", ngram_range=(3, 5))
        tfidf_tr = tfidf.fit_transform(urls_np[tr_idx]).toarray()
        tfidf_te = tfidf.transform(urls_np[te_idx]).toarray()
        
        # 2. Combine handcrafted + TF-IDF
        X_tr = np.hstack([X_tr_base.to_numpy(), tfidf_tr])
        X_te = np.hstack([X_te_base.to_numpy(), tfidf_te])
        
        # 3. Dynamic scale_pos_weight
        neg = (y_tr == 0).sum()
        pos = (y_tr == 1).sum()
        spw = round(neg / pos, 4)
        
        # 4. Strict Fold-Isolated StandardScaler (for SVM and LR)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42, class_weight="balanced")
        rf.fit(X_tr, y_tr)
        rf_preds[te_idx] = rf.predict(X_te)
        
        # XGBoost
        xgb_clf = xgb.XGBClassifier(n_estimators=300, eval_metric="logloss", n_jobs=-1, random_state=42, verbosity=0, scale_pos_weight=spw)
        xgb_clf.fit(X_tr, y_tr)
        xgb_preds[te_idx] = xgb_clf.predict(X_te)
        
        # SVM
        svm_clf = CalibratedClassifierCV(LinearSVC(dual=False, max_iter=2000, random_state=42, class_weight="balanced"), cv=3)
        svm_clf.fit(X_tr_scaled, y_tr)
        svm_preds[te_idx] = svm_clf.predict(X_te_scaled)
        
        # Logistic Regression
        try:
            from src.logistic_scratch import LogisticRegressionScratch
            lr = LogisticRegressionScratch(learning_rate=0.1, num_iterations=200)
            lr.fit(X_tr_scaled, y_tr)
            lr_preds[te_idx] = lr.predict(X_te_scaled)
        except Exception:
            pass

    print(f"     CV evaluation completed in {time.time()-t0:.1f}s")
    
    results.append(evaluate_model(y_np, rf_preds, "Random Forest (5-fold CV)"))
    plot_confusion_matrix(y_np, rf_preds, "Random Forest", GRAPHS_DIR)
    
    results.append(evaluate_model(y_np, xgb_preds, "XGBoost (5-fold CV)"))
    plot_confusion_matrix(y_np, xgb_preds, "XGBoost", GRAPHS_DIR)
    
    results.append(evaluate_model(y_np, svm_preds, "SVM Linear (5-fold CV)"))
    plot_confusion_matrix(y_np, svm_preds, "SVM", GRAPHS_DIR)
    
    if np.any(lr_preds):
        results.append(evaluate_model(y_np, lr_preds, "Logistic Reg. Scratch (5-fold CV)"))
        plot_confusion_matrix(y_np, lr_preds, "Logistic Reg. (Scratch)", GRAPHS_DIR)

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
    
    # Check if raw url column was retained (needed for fold-isolated TF-IDF)
    if "url" not in df_feat.columns:
        print("  Reloading raw URLs for strict cross-validation TF-IDF fitting...")
        df_urls = pd.read_csv(URLS_CSV)
        urls = clean_data(df_urls)["url"]
    else:
        urls = df_feat["url"]
        
    X = df_feat.drop(["url", "label"], axis=1, errors='ignore')
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
        results    = run_cross_validation(X, y, urls, n_splits=args.cv_folds)
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
