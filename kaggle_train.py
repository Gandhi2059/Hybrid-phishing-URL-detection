"""
kaggle_train.py
----------------
ShieldPhish — Full Dataset Training Script for Kaggle
Upload this file to Kaggle as a new Notebook (Script mode).

BEFORE RUNNING:
  1. Add your dataset under the Input section on the right panel
  2. Update DATASET_SLUG below to match your dataset folder name
     (run  print(os.listdir('/kaggle/input'))  to find it)
"""

import os, math, re, time, pickle, subprocess
subprocess.run(["pip", "install", "xgboost", "--quiet"], check=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — UPDATE THIS
# ─────────────────────────────────────────────────────────────────────────────
print("Datasets available:", os.listdir("/kaggle/input"))

DATASET_SLUG = "phishing-urls-raw"          # <-- change to your folder name
URLS_CSV     = f"/kaggle/input/{DATASET_SLUG}/urls.csv"
OUT_DIR      = "/kaggle/working/models"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load Dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[1/5] Loading dataset...")
print("="*60)
t0 = time.time()
df = pd.read_csv(URLS_CSV)
df = df.dropna(subset=["url"])
df["url"]   = df["url"].astype(str)
df["label"] = df["label"].astype(int)

NEG = int((df["label"] == 0).sum())   # legitimate
POS = int((df["label"] == 1).sum())   # phishing
SPW = round(NEG / POS, 4)             # XGBoost scale_pos_weight

print(f"  Rows      : {len(df):,}  ({time.time()-t0:.1f}s)")
print(f"  Legit  (0): {NEG:,}  ({NEG/len(df)*100:.1f}%)")
print(f"  Phish  (1): {POS:,}  ({POS/len(df)*100:.1f}%)")
print(f"  Ratio 0:1  = {NEG/POS:.2f}:1  →  XGBoost scale_pos_weight = {SPW}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Feature Engineering  (mirrors src/feature_engineering.py)
# ─────────────────────────────────────────────────────────────────────────────
SUSPICIOUS_KEYWORDS = [
    "login","verify","secure","bank","account",
    "update","sign","confirm","wallet","password","credential",
]
_KW_PAT  = "|".join(SUSPICIOUS_KEYWORDS)
_SPECIAL = r"[@?&=_%;]"
_IP      = r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}"
MAX_TFIDF = 100

def calculate_entropy(text: str) -> float:
    if not text:
        return 0.0
    return -sum(
        (p := text.count(c) / len(text)) * math.log2(p)
        for c in set(text)
    )

def extract_features(urls: pd.Series) -> pd.DataFrame:
    feats = pd.DataFrame({
        "url_length":       urls.str.len(),
        "num_dots":         urls.str.count(r"\."),
        "num_slashes":      urls.str.count("/"),
        "num_digits":       urls.str.count(r"\d"),
        "num_hyphens":      urls.str.count("-"),
        "num_special":      urls.str.count(_SPECIAL),
        "has_https":        urls.str.startswith("https://").astype(int),
        "has_ip":           urls.str.contains(_IP, regex=True).astype(int),
        "suspicious_words": urls.str.lower().str.count(_KW_PAT),
    }, index=urls.index)
    L = feats["url_length"].to_numpy(float)
    feats["digit_ratio"]   = feats["num_digits"].to_numpy(float)  / (L + 1e-5)
    feats["special_ratio"] = feats["num_special"].to_numpy(float) / (L + 1e-5)
    feats["entropy"]       = [calculate_entropy(u) for u in urls]
    return feats

print("\n[2/5] Extracting lexical + statistical features...")
t0  = time.time()
lex = extract_features(df["url"])
print(f"  Done in {time.time()-t0:.1f}s  →  {lex.shape[1]} features")

print("[3/5] Fitting & transforming TF-IDF (char 3-5 grams, 100 features)...")
t0    = time.time()
tfidf = TfidfVectorizer(max_features=MAX_TFIDF, analyzer="char", ngram_range=(3, 5))
tfidf.fit(df["url"].tolist())
tfidf_arr = tfidf.transform(df["url"].tolist()).toarray()
tfidf_df  = pd.DataFrame(
    tfidf_arr,
    columns=[f"tfidf_{i}" for i in range(MAX_TFIDF)],
    index=df.index,
)
print(f"  Done in {time.time()-t0:.1f}s")

X = pd.concat([lex, tfidf_df], axis=1)
y = df["label"]
print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

with open(f"{OUT_DIR}/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
print("  TF-IDF vectorizer saved.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — 5-Fold Stratified Cross-Validation
# Imbalance fix: class_weight='balanced' (RF/SVM), scale_pos_weight (XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Running 5-fold stratified cross-validation...")
cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []
cv_preds   = {}

def run_cv(model, name: str) -> None:
    print(f"\n  ▶ {name}...")
    t0    = time.time()
    preds = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    r = {
        "Model":     name,
        "Accuracy":  accuracy_score(y, preds),
        "Precision": precision_score(y, preds, zero_division=0),
        "Recall":    recall_score(y, preds, zero_division=0),
        "F1-Score":  f1_score(y, preds, zero_division=0),
        "F1-Macro":  f1_score(y, preds, average="macro", zero_division=0),
    }
    print(f"     Time={time.time()-t0:.1f}s | Acc={r['Accuracy']:.4f} | "
          f"Recall={r['Recall']:.4f} | F1={r['F1-Score']:.4f}")
    cv_results.append(r)
    cv_preds[name] = preds

# Random Forest — class_weight='balanced'
run_cv(
    RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42,
        class_weight="balanced",        # handles 3.4:1 imbalance
    ),
    "Random Forest",
)

# XGBoost — scale_pos_weight (XGBoost's equivalent of class_weight='balanced')
run_cv(
    xgb.XGBClassifier(
        n_estimators=300, eval_metric="logloss",
        n_jobs=-1, random_state=42, verbosity=0,
        scale_pos_weight=SPW,           # handles 3.4:1 imbalance
    ),
    "XGBoost",
)

# SVM — class_weight='balanced'
run_cv(
    Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            LinearSVC(
                dual=False, max_iter=2000, random_state=42,
                class_weight="balanced",  # handles 3.4:1 imbalance
            ),
            cv=3,
        )),
    ]),
    "SVM (LinearSVC)",
)

# Summary table
res_df = pd.DataFrame(cv_results)
print("\n" + "="*72)
print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} "
      f"{'F1':>8} {'F1-Macro':>9}")
print("="*72)
for _, row in res_df.iterrows():
    print(f"  {row['Model']:<20} {row['Accuracy']:>9.4f} {row['Precision']:>10.4f}"
          f" {row['Recall']:>8.4f} {row['F1-Score']:>8.4f} {row['F1-Macro']:>9.4f}")
print("="*72)
res_df.to_csv("/kaggle/working/cv_results.csv", index=False)

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, preds) in zip(axes, cv_preds.items()):
    cm = confusion_matrix(y, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legit", "Phishing"],
                yticklabels=["Legit", "Phishing"])
    ax.set_title(f"{name}\nAcc={accuracy_score(y,preds):.4f} | F1={f1_score(y,preds):.4f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.suptitle("5-Fold CV | 957K URLs | class_weight=balanced", y=1.02)
plt.tight_layout()
plt.savefig("/kaggle/working/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("Confusion matrices saved.")

# Metrics bar chart
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
x, w    = np.arange(len(metrics)), 0.25
fig, ax = plt.subplots(figsize=(12, 6))
colors  = ["#4C72B0", "#DD8452", "#55A868"]
for i, (_, row) in enumerate(res_df.iterrows()):
    vals = [row[m] for m in metrics]
    bars = ax.bar(x + i*w, vals, w, label=row["Model"], color=colors[i], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x + w)
ax.set_xticklabels(metrics)
ax.set_ylim(0.80, 1.01)
ax.set_title("Model Comparison | class_weight=balanced")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("/kaggle/working/metrics_comparison.png", dpi=150)
plt.close()
print("Metrics chart saved.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Train Final Models on 100% Data & Save
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Training final models on 100% of data...")

for model, name, fname in [
    (
        RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42,
            class_weight="balanced",
        ),
        "Random Forest", "rf_model.pkl",
    ),
    (
        xgb.XGBClassifier(
            n_estimators=300, eval_metric="logloss",
            n_jobs=-1, random_state=42, verbosity=0,
            scale_pos_weight=SPW,
        ),
        "XGBoost", "xgb_model.pkl",
    ),
    (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(
                    dual=False, max_iter=2000, random_state=42,
                    class_weight="balanced",
                ),
                cv=3,
            )),
        ]),
        "SVM (LinearSVC)", "svm_model.pkl",
    ),
]:
    print(f"  ▶ {name}...")
    t0 = time.time()
    model.fit(X, y)
    with open(f"{OUT_DIR}/{fname}", "wb") as f:
        pickle.dump(model, f)
    print(f"     ✔ Saved in {time.time()-t0:.1f}s → {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# Done — list all output files
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("ALL DONE — Download these from the Output tab:")
print("="*55)
for root, _, files in os.walk("/kaggle/working"):
    for fname in sorted(files):
        fpath = os.path.join(root, fname)
        print(f"  {fpath}  ({os.path.getsize(fpath)/1e6:.1f} MB)")
print("\nCopy all .pkl files → your local  models/  folder.")
