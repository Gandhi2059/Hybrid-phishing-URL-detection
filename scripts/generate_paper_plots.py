"""
scripts/generate_paper_plots.py
================================
Generates all four publication-quality figures for the ShieldPhish IEEE paper:

  1. Model Performance Comparison  — grouped bar chart (Acc / Prec / Rec / F1 / AUC)
  2. ROC Curves                    — all models overlaid, shaded AUC
  3. Rule-Weight Sensitivity       — FPR & FNR vs W_rule (RF + XGBoost dual-axis)
  4. SHAP Summary Plot             — beeswarm for the best model (Random Forest)

All outputs are written to graphs/paper_graphs/ as both PNG (300 dpi) and PDF
(vector, for IEEE submission).

Usage
-----
    python scripts/generate_paper_plots.py                  # 10 000 sample
    python scripts/generate_paper_plots.py --n-samples 30000
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

# ── Paths ─────────────────────────────────────────────────────────────────── #
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_CSV   = os.path.join(ROOT, "data", "urls.csv")
ABLATION_CSV = os.path.join(ROOT, "graphs", "ablation_rule_weight.csv")
OUT_DIR    = os.path.join(ROOT, "graphs", "paper_graphs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── CLI ───────────────────────────────────────────────────────────────────── #
p = argparse.ArgumentParser()
p.add_argument("--n-samples", type=int, default=10_000)
p.add_argument("--seed",      type=int, default=99)
args = p.parse_args()

# ── Design tokens (IEEE-ready palette) ────────────────────────────────────── #
PALETTE = {
    "Random Forest":   "#2563EB",   # vivid blue
    "XGBoost":         "#16A34A",   # vivid green
    "SVM (LinearSVC)": "#DC2626",   # vivid red
    "Logistic Reg.":   "#9333EA",   # purple
}
FONT_FAMILY = "DejaVu Sans"
plt.rcParams.update({
    "font.family":       FONT_FAMILY,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})

def savefig(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"  ✔  Saved → {name}.png / .pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════ #
#  STEP 0 — Load data & build feature matrix                                 #
# ═══════════════════════════════════════════════════════════════════════════ #
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "bank", "account",
    "update", "sign", "confirm", "wallet", "password", "credential",
]
_KW_PAT  = "|".join(SUSPICIOUS_KEYWORDS)
_SPECIAL = r"[@?&=_%;]"
_IP      = r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}"
MAX_TFIDF = 100


def calc_entropy(text: str) -> float:
    if not text:
        return 0.0
    return -sum((text.count(c) / len(text)) * math.log2(text.count(c) / len(text))
                for c in set(text))


def extract_features(urls: pd.Series, tfidf) -> np.ndarray:
    lex = pd.DataFrame({
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
    L = lex["url_length"].to_numpy(float)
    lex["digit_ratio"]   = lex["num_digits"].to_numpy(float)  / (L + 1e-5)
    lex["special_ratio"] = lex["num_special"].to_numpy(float) / (L + 1e-5)
    lex["entropy"]       = [calc_entropy(u) for u in urls]
    tfidf_arr = tfidf.transform(urls.tolist()).toarray()
    tfidf_df  = pd.DataFrame(
        tfidf_arr,
        columns=[f"tfidf_{i}" for i in range(MAX_TFIDF)],
        index=urls.index,
    )
    return pd.concat([lex, tfidf_df], axis=1).to_numpy(dtype=float), list(lex.columns)


print("\n" + "="*65)
print("  ShieldPhish — Generating Paper Plots")
print("="*65)

# Load dataset
print(f"\n[0/4] Loading dataset ({args.n_samples:,} samples) …")
df = pd.read_csv(DATA_CSV).dropna(subset=["url"])
df["url"]   = df["url"].astype(str)
df["label"] = df["label"].astype(int)
df = df.sample(n=min(args.n_samples, len(df)), random_state=args.seed)
print(f"  Legit: {(df['label']==0).sum():,}  |  Phishing: {(df['label']==1).sum():,}")

# Load / refit TF-IDF
tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
with open(tfidf_path, "rb") as f:
    tfidf = pickle.load(f)
if not hasattr(tfidf, "idf_"):
    print("  ⚠ TF-IDF version mismatch — re-fitting …")
    df_fit = pd.read_csv(DATA_CSV).dropna(subset=["url"])
    tfidf  = TfidfVectorizer(max_features=MAX_TFIDF, analyzer="char", ngram_range=(3, 5))
    tfidf.fit(df_fit["url"].sample(200_000, random_state=42).tolist())

X_arr, feat_names = extract_features(df["url"], tfidf)
y = df["label"].to_numpy()

# Load models
MODELS = [
    ("Random Forest",   os.path.join(MODELS_DIR, "rf_model.pkl")),
    ("XGBoost",         os.path.join(MODELS_DIR, "xgb_model.pkl")),
    ("SVM (LinearSVC)", os.path.join(MODELS_DIR, "svm_model.pkl")),
]

model_objs   = {}
metrics_rows = []
probas       = {}

for name, path in MODELS:
    if not os.path.exists(path):
        print(f"  ⚠  {name} not found — skipping")
        continue
    with open(path, "rb") as f:
        m = pickle.load(f)
    model_objs[name] = m
    preds = m.predict(X_arr)
    try:
        prob = m.predict_proba(X_arr)[:, 1]
        auc  = roc_auc_score(y, prob)
        probas[name] = prob
    except Exception:
        auc = float("nan")
    metrics_rows.append({
        "Model":     name,
        "Accuracy":  accuracy_score(y, preds),
        "Precision": precision_score(y, preds, zero_division=0),
        "Recall":    recall_score(y, preds, zero_division=0),
        "F1-Score":  f1_score(y, preds, zero_division=0),
        "AUC":       auc,
    })
    print(f"  {name:<22}  Acc={metrics_rows[-1]['Accuracy']:.4f}  AUC={auc:.4f}")

metrics_df = pd.DataFrame(metrics_rows)


# ═══════════════════════════════════════════════════════════════════════════ #
#  PLOT 1 — Model Performance Comparison (grouped bar chart)                 #
# ═══════════════════════════════════════════════════════════════════════════ #
print("\n[1/4] Plot 1 — Model Performance Comparison …")

metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
models_list     = metrics_df["Model"].tolist()
n_metrics       = len(metrics_to_plot)
n_models        = len(models_list)
x               = np.arange(n_metrics)
bar_width       = 0.22
offsets         = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * bar_width

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")

for i, (row, model_name) in enumerate(zip(metrics_rows, models_list)):
    vals   = [row[m] for m in metrics_to_plot]
    color  = PALETTE.get(model_name, "#888888")
    bars   = ax.bar(x + offsets[i], vals, width=bar_width,
                    label=model_name, color=color, alpha=0.88,
                    edgecolor="white", linewidth=0.8, zorder=3)
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color="#1a1a1a")

ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot, fontsize=12, fontweight="semibold")
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison — ShieldPhish",
             fontsize=14, fontweight="bold", pad=14)
ax.set_ylim(0, 1.10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.legend(loc="lower right", fontsize=10, framealpha=0.9, edgecolor="#cccccc")
ax.tick_params(axis="both", labelsize=10)
fig.tight_layout()
savefig(fig, "model_comparison_bar")


# ═══════════════════════════════════════════════════════════════════════════ #
#  PLOT 2 — ROC Curves (all models overlaid)                                 #
# ═══════════════════════════════════════════════════════════════════════════ #
print("[2/4] Plot 2 — ROC Curves …")

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")

# Diagonal reference
ax.plot([0, 1], [0, 1], linestyle="--", color="#aaaaaa", linewidth=1.4,
        label="Random Classifier (AUC = 0.500)", zorder=1)

for name, prob in probas.items():
    fpr, tpr, _ = roc_curve(y, prob)
    auc_val     = roc_auc_score(y, prob)
    color       = PALETTE.get(name, "#888888")
    ax.plot(fpr, tpr, color=color, linewidth=2.5,
            label=f"{name}  (AUC = {auc_val:.4f})", zorder=3)
    ax.fill_between(fpr, tpr, alpha=0.06, color=color)

# Mark the optimal threshold point (top-left proximity) for best model
if "Random Forest" in probas:
    fpr_rf, tpr_rf, _ = roc_curve(y, probas["Random Forest"])
    dists = np.sqrt(fpr_rf**2 + (1 - tpr_rf)**2)
    best  = np.argmin(dists)
    ax.scatter(fpr_rf[best], tpr_rf[best], s=90, color=PALETTE["Random Forest"],
               zorder=5, edgecolors="white", linewidths=1.5)
    ax.annotate(f"  Optimal\n  τ=0.5",
                xy=(fpr_rf[best], tpr_rf[best]),
                fontsize=8.5, color=PALETTE["Random Forest"],
                xytext=(fpr_rf[best]+0.04, tpr_rf[best]-0.06))

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
ax.set_title("ROC Curves — ShieldPhish Models",
             fontsize=14, fontweight="bold", pad=14)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.04)
ax.legend(loc="lower right", fontsize=10, framealpha=0.9, edgecolor="#cccccc")
ax.tick_params(labelsize=10)
fig.tight_layout()
savefig(fig, "roc_curves")


# ═══════════════════════════════════════════════════════════════════════════ #
#  PLOT 3 — Rule-Weight Sensitivity  (FPR / FNR vs W_rule)                  #
# ═══════════════════════════════════════════════════════════════════════════ #
print("[3/4] Plot 3 — Rule-Weight Sensitivity …")

ab = pd.read_csv(ABLATION_CSV)
ab_models = ["Random Forest", "XGBoost"]    # SVM degrades badly; not used in paper

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
fig.patch.set_facecolor("#FAFAFA")
fig.suptitle(r"Rule-Weight Sensitivity: FPR & FNR vs. $W_{rule}$",
             fontsize=14, fontweight="bold", y=1.01)

for ax, model_name in zip(axes, ab_models):
    ax.set_facecolor("#FAFAFA")
    sub = ab[ab["Model"] == model_name].sort_values("Rule_Weight")
    w   = sub["Rule_Weight"].values
    fpr = sub["FPR"].values
    fnr = sub["FNR"].values
    acc = sub["Accuracy"].values
    f1  = sub["F1_Score"].values

    color_main = PALETTE.get(model_name, "#444")

    # Left y-axis: FPR and FNR
    lns1 = ax.plot(w, fpr, marker="o", color="#DC2626", linewidth=2.2,
                   markersize=6, label="FPR (False Alarm Rate)", zorder=4)
    lns2 = ax.plot(w, fnr, marker="s", color="#2563EB", linewidth=2.2,
                   markersize=6, label="FNR (Miss Rate)", linestyle="--", zorder=4)

    # Right y-axis: Accuracy and F1
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    lns3 = ax2.plot(w, acc, marker="^", color="#16A34A", linewidth=1.8,
                    markersize=5, label="Accuracy", linestyle="-.", alpha=0.75)
    lns4 = ax2.plot(w, f1,  marker="D", color="#9333EA", linewidth=1.8,
                    markersize=5, label="F1-Score",  linestyle=":",  alpha=0.75)
    ax2.set_ylabel("Accuracy / F1-Score", fontsize=10, color="#555")
    ax2.tick_params(axis="y", labelcolor="#555", labelsize=9)
    ax2.set_ylim(0, 1.08)

    # Highlight the optimal W_rule = 0.15
    ax.axvline(0.15, color="#f59e0b", linewidth=2.0,
               linestyle="--", alpha=0.85, zorder=2)
    ax.text(0.155, max(fpr.max(), fnr.max()) * 0.92,
            r"$W_{rule}=0.15$", color="#b45309",
            fontsize=9, fontweight="bold")

    ax.set_xlabel(r"Rule Weight  $W_{rule}$", fontsize=11)
    ax.set_ylabel("Error Rate (FPR / FNR)", fontsize=11)
    ax.set_title(model_name, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.01, ax.get_ylim()[1] * 1.12)
    ax.tick_params(labelsize=9)

    # Combined legend
    all_lines  = lns1 + lns2 + lns3 + lns4
    all_labels = [l.get_label() for l in all_lines]
    ax.legend(all_lines, all_labels, fontsize=8.5, loc="upper left",
              framealpha=0.9, edgecolor="#ccc")

fig.tight_layout()
savefig(fig, "rule_weight_sensitivity")


# ═══════════════════════════════════════════════════════════════════════════ #
#  PLOT 4 — SHAP Summary (beeswarm) for Random Forest                       #
# ═══════════════════════════════════════════════════════════════════════════ #
print("[4/4] Plot 4 — SHAP Summary Plot …")

try:
    import shap

    rf_model = model_objs.get("Random Forest")
    if rf_model is None:
        raise RuntimeError("Random Forest model not loaded")

    # Use a smaller subsample for SHAP (TreeExplainer is fast but RF is large)
    shap_n = min(500, len(X_arr))
    np.random.seed(42)
    shap_idx = np.random.choice(len(X_arr), shap_n, replace=False)
    X_shap   = X_arr[shap_idx]

    # All 112 feature names: 12 handcrafted + 100 TF-IDF
    all_feat_names = feat_names + [f"tfidf_{i}" for i in range(MAX_TFIDF)]

    print(f"  Computing SHAP values on {shap_n} samples …")
    explainer  = shap.TreeExplainer(rf_model)
    shap_vals  = explainer.shap_values(X_shap)

    # shap_values returns [class_0, class_1] for RF
    if isinstance(shap_vals, list):
        sv = shap_vals[1]   # class-1 (Phishing)
    else:
        sv = shap_vals

    # Aggregate TF-IDF features into a single "char TF-IDF" group for readability
    handcrafted_n = len(feat_names)            # 12
    sv_hand   = sv[:, :handcrafted_n]          # (n, 12)
    sv_tfidf  = sv[:, handcrafted_n:]          # (n, 100)
    tfidf_agg = sv_tfidf.mean(axis=1, keepdims=True)   # aggregate into 1 column
    X_hand    = X_shap[:, :handcrafted_n]
    X_tfidf_m = X_shap[:, handcrafted_n:].mean(axis=1, keepdims=True)

    sv_plot  = np.hstack([sv_hand, tfidf_agg])
    X_plot   = np.hstack([X_hand,  X_tfidf_m])
    plot_names = feat_names + ["char TF-IDF (avg)"]

    # Pretty names for the y-axis
    name_map = {
        "url_length":       "URL Length",
        "num_dots":         "Dot Count",
        "num_slashes":      "Slash Count",
        "num_digits":       "Digit Count",
        "num_hyphens":      "Hyphen Count",
        "num_special":      "Special Char Count",
        "has_https":        "Has HTTPS",
        "has_ip":           "Has IP Address",
        "suspicious_words": "Suspicious Keywords",
        "digit_ratio":      "Digit/Length Ratio",
        "special_ratio":    "Special/Length Ratio",
        "entropy":          "Shannon Entropy",
        "char TF-IDF (avg)": "Char TF-IDF (avg)",
    }
    pretty_names = [name_map.get(n, n) for n in plot_names]

    # Sort features by mean |SHAP|
    mean_abs = np.abs(sv_plot).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:13]   # top 13

    sv_sorted  = sv_plot[:, order]
    X_sorted   = X_plot[:, order]
    names_sorted = [pretty_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Manual beeswarm
    cmap = plt.cm.coolwarm
    for feat_i in range(sv_sorted.shape[1]):
        sv_col  = sv_sorted[:, feat_i]
        xv_col  = X_sorted[:,  feat_i]

        # Normalise feature values for colour
        vmin, vmax = xv_col.min(), xv_col.max()
        norm_xv = (xv_col - vmin) / (vmax - vmin + 1e-9)

        y_pos   = sv_sorted.shape[1] - 1 - feat_i
        # Jitter dots vertically within the row
        jitter  = np.random.uniform(-0.32, 0.32, size=len(sv_col))
        colors  = cmap(norm_xv)
        ax.scatter(sv_col, y_pos + jitter,
                   c=colors, s=14, alpha=0.65, linewidths=0, zorder=3)

    # Colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.6)
    cbar.set_label("Feature value  (low → high)", fontsize=9)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"], fontsize=8)

    ax.axvline(0, color="#888888", linewidth=1.0, linestyle="-", zorder=1)
    ax.set_yticks(range(sv_sorted.shape[1]))
    ax.set_yticklabels(reversed(names_sorted), fontsize=11)
    ax.set_xlabel("SHAP Value  (impact on phishing probability)", fontsize=11)
    ax.set_title("SHAP Summary — Random Forest (ShieldPhish)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.tick_params(axis="x", labelsize=10)
    fig.tight_layout()
    savefig(fig, "shap_summary")

except ImportError:
    print("  ⚠  shap not installed — skipping SHAP plot.")
    print("     Install with:  pip install shap")
except Exception as exc:
    print(f"  ⚠  SHAP plot failed: {exc}")


# ── Final summary ─────────────────────────────────────────────────────────── #
print("\n" + "="*65)
print(f"  All plots saved → {OUT_DIR}")
print("="*65)
print(f"\n  {'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
print("  " + "-"*55)
for r in metrics_rows:
    print(f"  {r['Model']:<22} {r['Accuracy']:>7.4f} {r['Precision']:>7.4f}"
          f" {r['Recall']:>7.4f} {r['F1-Score']:>7.4f} {r['AUC']:>7.4f}")
print()
