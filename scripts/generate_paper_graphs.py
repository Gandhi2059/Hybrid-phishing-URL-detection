"""
scripts/generate_paper_graphs.py
---------------------------------
Generates the three missing publication-grade graphs for the IEEE paper:

  1. Model Performance Comparison  — grouped bar chart
     (Accuracy / Precision / Recall / F1-Score / ROC-AUC for RF, XGB, SVM)

  2. ROC Curves (all three models on a single axes + zoom inset)
     Saved as PNG + SVG + PDF.

  3. SHAP Summary Plot (beeswarm) + SHAP bar chart (mean |SHAP|)
     Uses shap.TreeExplainer on Random Forest (or XGBoost fallback).
     Saved as PNG + PDF.

All outputs go to  graphs/paper_graphs/  (separate from existing files).

Usage
-----
    python scripts/generate_paper_graphs.py                   # 10 000 samples
    python scripts/generate_paper_graphs.py --n-samples 30000
    python scripts/generate_paper_graphs.py --full
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ── Paths ─────────────────────────────────────────────────────────────────── #
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_CSV   = os.path.join(ROOT, "data", "urls.csv")
OUT_DIR    = os.path.join(ROOT, "graphs", "paper_graphs")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_TFIDF = 100

# ── Matplotlib style ──────────────────────────────────────────────────────── #
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi":      150,
    "savefig.dpi":     300,
})
PALETTE = ["#2196F3", "#FF5722", "#4CAF50"]   # blue / deep-orange / green

# ── CLI ───────────────────────────────────────────────────────────────────── #
p = argparse.ArgumentParser()
p.add_argument("--data",      default=DATA_CSV)
p.add_argument("--n-samples", type=int, default=10_000)
p.add_argument("--full",      action="store_true")
p.add_argument("--seed",      type=int, default=99)
args = p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Feature engineering — exact mirror of kaggle_train.py
# ══════════════════════════════════════════════════════════════════════════════
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "bank", "account",
    "update", "sign", "confirm", "wallet", "password", "credential",
]
_KW_PAT  = "|".join(SUSPICIOUS_KEYWORDS)
_SPECIAL = r"[@?&=_%;]"
_IP      = r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}"


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    return -sum((prob := text.count(c) / len(text)) * math.log2(prob) for c in set(text))


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
    feats["entropy"]       = [_entropy(u) for u in urls]
    return feats


def build_feature_matrix(urls: pd.Series, tfidf) -> tuple[np.ndarray, list]:
    lex      = extract_lexical(urls)
    tfidf_df = pd.DataFrame(
        tfidf.transform(urls.tolist()).toarray(),
        columns=[f"tfidf_{i}" for i in range(MAX_TFIDF)],
        index=urls.index,
    )
    combined = pd.concat([lex, tfidf_df], axis=1)
    return combined.to_numpy(dtype=float), list(combined.columns)


# ══════════════════════════════════════════════════════════════════════════════
# Load & sample data
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  ShieldPhish — Paper Graph Generator")
print("=" * 65)

df = pd.read_csv(args.data).dropna(subset=["url"])
df["url"]   = df["url"].astype(str)
df["label"] = df["label"].astype(int)
print(f"\n  Loaded {len(df):,} rows")

if not args.full:
    n  = min(args.n_samples, len(df))
    df = df.sample(n=n, random_state=args.seed)
    print(f"  Sampled {len(df):,} rows (seed={args.seed})")

y = df["label"].to_numpy()
print(f"  Legit: {(y==0).sum():,}  |  Phishing: {(y==1).sum():,}")

# ── TF-IDF ────────────────────────────────────────────────────────────────── #
tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
if not os.path.exists(tfidf_path):
    sys.exit(f"\n❌ TF-IDF vectorizer not found at {tfidf_path}")

with open(tfidf_path, "rb") as f:
    tfidf_loaded = pickle.load(f)

if hasattr(tfidf_loaded, "idf_"):
    tfidf = tfidf_loaded
else:
    print("\n  TF-IDF version mismatch — re-fitting …")
    df_all = pd.read_csv(args.data).dropna(subset=["url"])
    df_all["url"] = df_all["url"].astype(str)
    fit_sample = df_all["url"].sample(n=min(200_000, len(df_all)), random_state=42).tolist()
    tfidf = TfidfVectorizer(max_features=MAX_TFIDF, analyzer="char", ngram_range=(3, 5))
    tfidf.fit(fit_sample)

print("\n  Extracting features …")
X, feature_names = build_feature_matrix(df["url"], tfidf)
print(f"  Feature matrix: {X.shape[0]:,} x {X.shape[1]}")


# ══════════════════════════════════════════════════════════════════════════════
# Evaluate models
# ══════════════════════════════════════════════════════════════════════════════
MODELS_INFO = [
    ("Random Forest",   os.path.join(MODELS_DIR, "rf_model.pkl"),  PALETTE[0]),
    ("XGBoost",         os.path.join(MODELS_DIR, "xgb_model.pkl"), PALETTE[1]),
    ("SVM (LinearSVC)", os.path.join(MODELS_DIR, "svm_model.pkl"), PALETTE[2]),
]

results  = []
roc_data = []
shap_model, shap_model_name, shap_X, shap_feat_names = None, None, None, None

for name, path, color in MODELS_INFO:
    if not os.path.exists(path):
        print(f"\n  ⚠️  {name}: not found — skipping")
        continue

    print(f"\n  [{name}] loading …")
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict):
        model  = payload["model"]
        scaler = payload.get("scaler")
        feats  = getattr(scaler, "feature_names_in_",
                         getattr(model, "feature_names_in_", None))
        col_idx = [feature_names.index(c) for c in feats] if feats is not None else None
        X_ev    = X[:, col_idx] if col_idx is not None else X
        X_ev    = scaler.transform(X_ev) if scaler else X_ev
    else:
        model   = payload
        feats   = getattr(model, "feature_names_in_", None)
        col_idx = [feature_names.index(c) for c in feats] if feats is not None else None
        X_ev    = X[:, col_idx] if col_idx is not None else X

    preds = model.predict(X_ev)
    acc   = accuracy_score(y, preds)
    prec  = precision_score(y, preds, zero_division=0)
    rec   = recall_score(y, preds, zero_division=0)
    f1    = f1_score(y, preds, zero_division=0)

    auc_val = float("nan")
    try:
        proba   = model.predict_proba(X_ev)[:, 1]
        auc_val = roc_auc_score(y, proba)
        fpr_v, tpr_v, _ = roc_curve(y, proba)
        roc_data.append((fpr_v, tpr_v, auc_val, name, color))
    except Exception:
        pass

    print(f"    Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc_val:.4f}")
    results.append({"Model": name, "Accuracy": acc, "Precision": prec,
                    "Recall": rec, "F1-Score": f1, "ROC-AUC": auc_val})

    if name in ("Random Forest", "XGBoost") and shap_model is None:
        shap_model      = model
        shap_model_name = name
        shap_X          = X_ev
        shap_feat_names = ([feature_names[i] for i in col_idx]
                           if col_idx is not None else feature_names)

results_df = pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 1 — Grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n  [GRAPH 1] Model Performance Comparison Bar Chart …")

metrics   = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
n_models  = len(results_df)
bar_w     = 0.18
x         = np.arange(len(metrics))

fig, ax = plt.subplots(figsize=(12, 6))
for i, (_, row) in enumerate(results_df.iterrows()):
    vals    = [row[m] for m in metrics]
    offsets = x + (i - n_models / 2 + 0.5) * bar_w
    bars    = ax.bar(offsets, vals, bar_w,
                     label=row["Model"],
                     color=MODELS_INFO[i][2],
                     alpha=0.88, edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.09)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison — RF vs XGBoost vs SVM",
             fontweight="bold", pad=12)
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(axis="y", linestyle=":", alpha=0.45)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
for ext in ("png", "svg", "pdf"):
    fig.savefig(os.path.join(OUT_DIR, f"model_comparison_bar.{ext}"), bbox_inches="tight")
plt.close(fig)
print(f"  ✅ model_comparison_bar.{{png,svg,pdf}} → {OUT_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 2 — ROC Curves
# ══════════════════════════════════════════════════════════════════════════════
print("\n  [GRAPH 2] ROC Curves …")

if roc_data:
    fig, ax = plt.subplots(figsize=(8, 7))
    for fpr_v, tpr_v, auc_v, name, color in roc_data:
        ax.plot(fpr_v, tpr_v, lw=2.2, color=color,
                label=f"{name}  (AUC = {auc_v:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.45, label="Random classifier")

    ax.set_xlabel("False Positive Rate  (1 − Specificity)")
    ax.set_ylabel("True Positive Rate  (Sensitivity / Recall)")
    ax.set_title("ROC Curves — Phishing URL Detection", fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # Zoom inset — upper-left, showing the near-perfect corner
    try:
        axins = ax.inset_axes([0.08, 0.55, 0.38, 0.38])
        for fpr_v, tpr_v, _, name, color in roc_data:
            axins.plot(fpr_v, tpr_v, lw=1.5, color=color)
        axins.set_xlim(0, 0.08)
        axins.set_ylim(0.92, 1.0)
        axins.set_title("Zoomed", fontsize=8)
        axins.tick_params(labelsize=7)
        axins.grid(True, linestyle=":", alpha=0.4)
        ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.5)
    except Exception:
        pass   # indicate_inset_zoom needs matplotlib >= 3.3

    plt.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"roc_curves.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ roc_curves.{{png,svg,pdf}} → {OUT_DIR}/")
else:
    print("  ⚠️  No ROC data — all models missing predict_proba.")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 3 — SHAP Summary Plot
# ══════════════════════════════════════════════════════════════════════════════
print("\n  [GRAPH 3] SHAP Summary Plot …")

try:
    import shap

    SHAP_N     = min(2_000, shap_X.shape[0])
    rng        = np.random.default_rng(42)
    idx        = rng.choice(shap_X.shape[0], size=SHAP_N, replace=False)
    X_shap     = shap_X[idx]

    print(f"  Using {shap_model_name}  |  {SHAP_N} samples  |  {len(shap_feat_names)} features")
    explainer = shap.TreeExplainer(shap_model)
    shap_vals = explainer.shap_values(X_shap)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]   # phishing class

    # — Beeswarm ——————————————————————————————————————————————————————————— #
    shap.summary_plot(shap_vals, X_shap,
                      feature_names=shap_feat_names,
                      max_display=20, plot_type="dot",
                      show=False, color_bar=True)
    fig_s = plt.gcf()
    fig_s.suptitle(
        f"SHAP Feature Importance — {shap_model_name}  (phishing class)",
        fontweight="bold", fontsize=13, y=1.01)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig_s.savefig(os.path.join(OUT_DIR, f"shap_summary.{ext}"),
                      bbox_inches="tight", dpi=300)
    plt.close(fig_s)
    print(f"  ✅ shap_summary.{{png,pdf}} → {OUT_DIR}/")

    # — Bar chart (mean |SHAP|) ————————————————————————————————————————————— #
    shap.summary_plot(shap_vals, X_shap,
                      feature_names=shap_feat_names,
                      max_display=20, plot_type="bar",
                      show=False)
    fig_b = plt.gcf()
    fig_b.suptitle(
        f"SHAP Mean |SHAP| — {shap_model_name}  (Top 20 features)",
        fontweight="bold", fontsize=13, y=1.01)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig_b.savefig(os.path.join(OUT_DIR, f"shap_bar.{ext}"),
                      bbox_inches="tight", dpi=300)
    plt.close(fig_b)
    print(f"  ✅ shap_bar.{{png,pdf}} → {OUT_DIR}/")

except ImportError:
    print("  ⚠️  shap not installed.  Run:  pip install shap")
except Exception as e:
    print(f"  ⚠️  SHAP failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Done. Outputs in:")
print(f"  {OUT_DIR}/")
print("    model_comparison_bar.{png,svg,pdf}")
print("    roc_curves.{png,svg,pdf}")
print("    shap_summary.{png,pdf}")
print("    shap_bar.{png,pdf}")
print("=" * 65 + "\n")
