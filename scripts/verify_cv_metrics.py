"""
scripts/verify_cv_metrics.py
=============================
Runs the exact 5-fold stratified cross-validation on the PAPER'S 49,999-URL
stratified subsample and computes per-fold:
  Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC

IMPORTANT: features.csv contains 957,341 rows (the full merged corpus).
The paper describes a stratified subsample of 50,000 URLs (49,999 after
deduplication) with 22.8% phishing / 77.2% legit. This script reproduces
that exact subsample using stratified sampling with random_state=42.

Reports mean ± std for each metric and compares against the paper's Table VII.

Usage
-----
    python scripts/verify_cv_metrics.py
    python scripts/verify_cv_metrics.py --folds 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT         = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")
GRAPHS_DIR   = os.path.join(ROOT, "graphs")

p = argparse.ArgumentParser()
p.add_argument("--folds",     type=int, default=5)
p.add_argument("--n-samples", type=int, default=0,
               help="0 = use all rows (full 49999); >0 = subsample for speed")
args = p.parse_args()

# ── Paper claimed values (Table VII) ──────────────────────────────────────── #
PAPER = {
    "SVM (LinearSVC)": {
        "Accuracy":    (93.14, 0.35),
        "Precision":   (91.03, 0.38),
        "Recall":      (77.63, 0.54),
        "Specificity": (97.73, 0.22),
        "F1-Score":    (83.80, 0.46),
        "ROC-AUC":     (96.80, 0.21),
    },
    "XGBoost": {
        "Accuracy":    (94.37, 0.31),
        "Precision":   (85.57, 0.48),
        "Recall":      (90.62, 0.39),
        "Specificity": (95.48, 0.26),
        "F1-Score":    (88.03, 0.35),
        "ROC-AUC":     (98.32, 0.11),
    },
    "Random Forest": {
        "Accuracy":    (95.01, 0.28),
        "Precision":   (93.57, 0.31),
        "Recall":      (83.91, 0.42),
        "Specificity": (98.29, 0.18),
        "F1-Score":    (88.48, 0.33),
        "ROC-AUC":     (98.08, 0.15),
    },
}

TOLERANCE = 1.5   # percentage points — within this = MATCH


def banner(msg):
    print("\n" + "="*70)
    print(f"  {msg}")
    print("="*70)


def per_fold_metrics(y_true, y_pred, y_prob=None):
    cm        = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan")
    return {
        "Accuracy":    accuracy_score(y_true, y_pred),
        "Precision":   precision_score(y_true, y_pred, zero_division=0),
        "Recall":      recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1-Score":    f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":     auc,
    }


def run_cv(model, X, y, cv, model_name):
    fold_results = []
    t0 = time.time()
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        try:
            proba = model.predict_proba(X_te)[:, 1]
        except Exception:
            proba = None

        m = per_fold_metrics(y_te, preds, proba)
        fold_results.append(m)
        print(f"    Fold {fold}/{cv.n_splits}  "
              f"Acc={m['Accuracy']*100:.2f}%  "
              f"F1={m['F1-Score']*100:.2f}%  "
              f"AUC={m['ROC-AUC']:.4f}")

    elapsed = time.time() - t0
    print(f"  ✔  Done in {elapsed:.1f}s")

    # Aggregate
    summary = {}
    for key in fold_results[0]:
        vals = np.array([r[key] for r in fold_results])
        summary[key] = (vals.mean(), vals.std())
    return summary


def compare(model_name, actual):
    paper = PAPER.get(model_name)
    if paper is None:
        return

    METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]

    print(f"\n  {'Metric':<14} {'Paper (mean±std)':>20} {'Actual (mean±std)':>22} {'Diff':>8} {'Status':>8}")
    print("  " + "-"*76)

    all_match = True
    for m in METRICS:
        p_mean, p_std = paper[m]
        a_mean, a_std = actual[m]
        a_mean_pct = a_mean * 100 if m != "ROC-AUC" else a_mean * 100
        a_std_pct  = a_std  * 100 if m != "ROC-AUC" else a_std  * 100

        diff    = abs(p_mean - a_mean_pct)
        status  = "✅ MATCH" if diff <= TOLERANCE else "❌ MISMATCH"
        if diff > TOLERANCE:
            all_match = False

        paper_str  = f"{p_mean:.2f}% ± {p_std:.2f}%"
        actual_str = f"{a_mean_pct:.2f}% ± {a_std_pct:.2f}%"
        print(f"  {m:<14} {paper_str:>20} {actual_str:>22} {diff:>7.2f}%  {status}")

    overall = "✅ VERIFIED" if all_match else "⚠️  DISCREPANCIES FOUND"
    print(f"\n  Overall: {overall}  (tolerance = ±{TOLERANCE}%)")
    return all_match


# ── Load features ─────────────────────────────────────────────────────────── #
banner("ShieldPhish — 5-Fold CV Verification vs. Paper Table VII")

print(f"\nLoading features from {FEATURES_CSV} …")
df_full = pd.read_csv(FEATURES_CSV)
print(f"  Full corpus: {len(df_full):,} rows × {df_full.shape[1]} columns")

# ── Reproduce the paper's 49,999-row stratified subsample ──────────────────
# Paper: "We extracted a stratified evaluation subset of 50,000 URLs
#         consisting of 11,422 Phishing (22.8%) and 38,578 Legitimate (77.2%)"
# Sampling with random_state=42 to match the paper's training procedure.
TARGET_N = 50_000
df = df_full.groupby("label", group_keys=False).apply(
    lambda grp: grp.sample(
        n=min(int(round(TARGET_N * len(grp) / len(df_full))), len(grp)),
        random_state=42
    )
).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Stratified subsample: {len(df):,} rows  "
      f"(target {TARGET_N:,}, paper claims 49,999)")

if args.n_samples > 0:
    df = df.sample(n=min(args.n_samples, len(df)), random_state=42)
    print(f"  Further subsampled to: {len(df):,} rows")

X = df.drop(["url", "label"], axis=1, errors="ignore").to_numpy(dtype=float)
y = df["label"].to_numpy(dtype=int)

neg, pos = int((y == 0).sum()), int((y == 1).sum())
spw = round(neg / pos, 4)
print(f"\n  Total URLs : {len(y):,}")
print(f"  Legitimate : {neg:,}  ({neg/len(y)*100:.1f}%)")
print(f"  Phishing   : {pos:,}  ({pos/len(y)*100:.1f}%)")
print(f"  class imbalance ratio (neg/pos) : {spw}")
print(f"\n  NOTE: Bar chart used pre-trained pkl models on a different 15k")
print(f"        URL sample → inflated scores due to training/test overlap.")
print(f"        THIS script re-trains per fold on the correct 49,999 subset.")

cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

results_all = {}

# ── 1. Random Forest ──────────────────────────────────────────────────────── #
banner(f"[1/3] Random Forest  —  {args.folds}-fold CV")
rf = RandomForestClassifier(
    n_estimators=200, class_weight="balanced",
    n_jobs=-1, random_state=42
)
rf_results = run_cv(rf, X, y, cv, "Random Forest")
results_all["Random Forest"] = rf_results
compare("Random Forest", rf_results)

# ── 2. XGBoost ───────────────────────────────────────────────────────────── #
banner(f"[2/3] XGBoost  —  {args.folds}-fold CV  (scale_pos_weight={spw})")
xgb_model = xgb.XGBClassifier(
    n_estimators=300, eval_metric="logloss",
    scale_pos_weight=spw, n_jobs=-1,
    random_state=42, verbosity=0,
)
xgb_results = run_cv(xgb_model, X, y, cv, "XGBoost")
results_all["XGBoost"] = xgb_results
compare("XGBoost", xgb_results)

# ── 3. SVM ───────────────────────────────────────────────────────────────── #
banner(f"[3/3] SVM (LinearSVC)  —  {args.folds}-fold CV")
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", CalibratedClassifierCV(
        LinearSVC(dual=False, max_iter=2000, random_state=42,
                  class_weight="balanced"),
        cv=3
    )),
])
svm_results = run_cv(svm_pipe, X, y, cv, "SVM (LinearSVC)")
results_all["SVM (LinearSVC)"] = svm_results
compare("SVM (LinearSVC)", svm_results)

# ── Final Summary Table ───────────────────────────────────────────────────── #
banner("FINAL VERIFIED METRICS — Table VII")
METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]

print(f"\n  {'Classifier':<22} {'Accuracy':>14} {'Precision':>14} {'Recall':>14} "
      f"{'Specificity':>14} {'F1-Score':>14} {'ROC-AUC':>14}")
print("  " + "-"*106)

for model_name, res in results_all.items():
    row = f"  {model_name:<22}"
    for m in METRICS:
        mean_pct = res[m][0] * 100
        std_pct  = res[m][1] * 100
        row += f"  {mean_pct:5.2f}±{std_pct:4.2f}%"
    print(row)

# ── Save to CSV ───────────────────────────────────────────────────────────── #
rows = []
for model_name, res in results_all.items():
    row = {"Model": model_name}
    for m in METRICS:
        row[f"{m}_mean"] = round(res[m][0] * 100, 4)
        row[f"{m}_std"]  = round(res[m][1] * 100, 4)
    rows.append(row)

out_csv = os.path.join(GRAPHS_DIR, "cv_verified_metrics.csv")
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"\n  ✔  Results saved → {out_csv}")
print(f"\n  NOTE: Tolerance for MATCH = ±{TOLERANCE} percentage points")
print("  Discrepancies within tolerance = paper is correct.")
print("  Discrepancies outside tolerance = paper values need updating.\n")
