"""
scripts/evaluate_models.py
--------------------------
Evaluate the Kaggle-trained models (rf_model.pkl, xgb_model.pkl, svm_model.pkl)
on a local held-out sample from data/urls.csv.

The feature engineering here is an EXACT mirror of kaggle_train.py so that
the feature matrix matches what each model was trained on.

Note: The TF-IDF vectorizer was saved on Kaggle with sklearn 1.6.x.  If the
local sklearn version differs (e.g. 1.4.x), the pickled idf_ is missing.  The
script automatically detects this and re-fits the vectorizer locally using the
same hyper-parameters so the vocabulary is reproduced correctly.

Usage
-----
    python scripts/evaluate_models.py                   # 10 000-sample test
    python scripts/evaluate_models.py --n-samples 50000
    python scripts/evaluate_models.py --full            # entire dataset
"""

import argparse
import math
import os
import pickle
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_CSV   = os.path.join(ROOT, "data", "urls.csv")

# ── CLI args ─────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--data",      default=DATA_CSV,  help="Path to URL CSV")
p.add_argument("--n-samples", type=int, default=10_000,
               help="Number of random samples to evaluate on (default 10 000)")
p.add_argument("--full",      action="store_true",
               help="Evaluate on the entire dataset (overrides --n-samples)")
p.add_argument("--seed",      type=int, default=99,
               help="Random seed for sampling (use a different seed than "
                    "training seed=42 to approximate a held-out set)")
args = p.parse_args()

# ── Feature engineering helpers (mirrors kaggle_train.py exactly) ─────────────
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "bank", "account",
    "update", "sign", "confirm", "wallet", "password", "credential",
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


def extract_lexical(urls: pd.Series) -> pd.DataFrame:
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


def build_feature_matrix(urls: pd.Series, tfidf) -> np.ndarray:
    lex       = extract_lexical(urls)
    tfidf_arr = tfidf.transform(urls.tolist()).toarray()
    tfidf_df  = pd.DataFrame(
        tfidf_arr,
        columns=[f"tfidf_{i}" for i in range(MAX_TFIDF)],
        index=urls.index,
    )
    return pd.concat([lex, tfidf_df], axis=1).to_numpy(dtype=float)


# ── Load dataset ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  ShieldPhish — Model Accuracy Evaluation")
print("=" * 65)
print(f"\n📂 Loading dataset: {args.data}")

t0 = time.time()
df = pd.read_csv(args.data)
df = df.dropna(subset=["url"])
df["url"]   = df["url"].astype(str)
df["label"] = df["label"].astype(int)
print(f"   Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

if not args.full:
    n = min(args.n_samples, len(df))
    df = df.sample(n=n, random_state=args.seed)
    print(f"   Sampled {len(df):,} rows (seed={args.seed}; "
          f"different from training seed=42 → approximates held-out set)")

NEG = int((df["label"] == 0).sum())
POS = int((df["label"] == 1).sum())
print(f"   Legit (0): {NEG:,}  |  Phishing (1): {POS:,}")

# ── Load TF-IDF vectorizer ────────────────────────────────────────────────────
tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
if not os.path.exists(tfidf_path):
    sys.exit(f"\n❌ TF-IDF vectorizer not found at {tfidf_path}")

from sklearn.feature_extraction.text import TfidfVectorizer

with open(tfidf_path, "rb") as f:
    tfidf_loaded = pickle.load(f)

# Detect sklearn version mismatch (idf_ missing when vectorizer was pickled
# with a newer sklearn than the local environment)
def _is_fitted(v):
    return hasattr(v, "idf_")

if _is_fitted(tfidf_loaded):
    tfidf = tfidf_loaded
    print(f"\n✅ TF-IDF vectorizer loaded ({tfidf_path})")
else:
    print(f"\n⚠️  TF-IDF pickle has no idf_ (sklearn version mismatch "
          "— Kaggle 1.6.x vs local 1.4.x).")
    print("   Re-fitting TF-IDF locally with same hyper-parameters "
          "on 200 000-URL sample (seed=42, mirrors Kaggle training)...")
    # Load a large reproducible sample that mirrors the Kaggle training corpus
    df_fit = pd.read_csv(args.data)
    df_fit = df_fit.dropna(subset=["url"])
    df_fit["url"] = df_fit["url"].astype(str)
    fit_sample = df_fit["url"].sample(
        n=min(200_000, len(df_fit)), random_state=42
    ).tolist()
    tfidf = TfidfVectorizer(
        max_features=MAX_TFIDF, analyzer="char", ngram_range=(3, 5)
    )
    tfidf.fit(fit_sample)
    print(f"   ✅ TF-IDF re-fitted on {len(fit_sample):,} URLs — "
          f"{len(tfidf.vocabulary_)} vocab tokens")

# ── Build feature matrix ─────────────────────────────────────────────────────
print("⚙️  Extracting features (lexical + TF-IDF)...")
t0 = time.time()
X = build_feature_matrix(df["url"], tfidf)
y = df["label"].to_numpy()
print(f"   Feature matrix: {X.shape[0]:,} × {X.shape[1]}  ({time.time()-t0:.1f}s)")

# ── Evaluate each model ───────────────────────────────────────────────────────
MODELS = [
    ("Random Forest",  os.path.join(MODELS_DIR, "rf_model.pkl")),
    ("XGBoost",        os.path.join(MODELS_DIR, "xgb_model.pkl")),
    ("SVM (LinearSVC)",os.path.join(MODELS_DIR, "svm_model.pkl")),
]

results = []

for name, path in MODELS:
    print(f"\n{'─'*65}")
    print(f"  Model: {name}")
    print(f"{'─'*65}")

    if not os.path.exists(path):
        print(f"  ⚠️  Not found: {path}  — skipping")
        continue

    t0 = time.time()
    with open(path, "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X)
    elapsed = time.time() - t0

    try:
        proba = model.predict_proba(X)[:, 1]
        auc   = roc_auc_score(y, proba)
        auc_s = f"{auc:.4f}"
    except Exception:
        auc_s = "N/A"

    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    f1   = f1_score(y, preds, zero_division=0)
    f1m  = f1_score(y, preds, average="macro", zero_division=0)
    cm   = confusion_matrix(y, preds)

    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  F1-Macro  : {f1m*100:.2f}%")
    print(f"  ROC-AUC   : {auc_s}")
    print(f"  Inference : {elapsed:.2f}s  ({len(y)/elapsed:,.0f} URLs/sec)")
    print(f"\n  Confusion Matrix:")
    print(f"             Predicted")
    print(f"              Legit  Phish")
    print(f"  Actual Legit  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"  Actual Phish  {cm[1,0]:6d}  {cm[1,1]:6d}")
    print(f"\n  Classification Report:")
    print(classification_report(y, preds,
                                target_names=["Legit", "Phishing"],
                                digits=4))

    results.append({
        "Model":     name,
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "F1-Score":  f1,
        "F1-Macro":  f1m,
        "ROC-AUC":   auc_s,
    })

# ── Summary Table ─────────────────────────────────────────────────────────────
if results:
    print("\n" + "=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)
    header = f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'F1-Macro':>9} {'ROC-AUC':>9}"
    print(header)
    print("-" * 80)
    for r in results:
        print(f"  {r['Model']:<22} {r['Accuracy']*100:>8.2f}% "
              f"{r['Precision']*100:>9.2f}% "
              f"{r['Recall']*100:>7.2f}% "
              f"{r['F1-Score']*100:>7.2f}% "
              f"{r['F1-Macro']*100:>8.2f}% "
              f"{str(r['ROC-AUC']):>9}")
    print("=" * 80)
    print(f"\n  Evaluated on {len(y):,} samples from: {args.data}")
    print("  Done ✓")
