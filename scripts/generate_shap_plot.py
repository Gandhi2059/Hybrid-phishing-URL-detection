"""
scripts/generate_shap_plot.py
==============================
Standalone SHAP summary (beeswarm) plot using XGBoost.
XGBoost is orders of magnitude faster than the 1.5GB RF for TreeExplainer.

Usage
-----
    python scripts/generate_shap_plot.py
    python scripts/generate_shap_plot.py --n-shap 300
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
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Paths ────────────────────────────────────────────────────────────────── #
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_CSV   = os.path.join(ROOT, "data", "urls.csv")
OUT_DIR    = os.path.join(ROOT, "graphs", "paper_graphs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── CLI ──────────────────────────────────────────────────────────────────── #
p = argparse.ArgumentParser()
p.add_argument("--n-shap",   type=int, default=200,
               help="Number of samples for SHAP (default 200 — fast)")
p.add_argument("--n-load",   type=int, default=5000,
               help="Rows to load for feature building (default 5000)")
p.add_argument("--seed",     type=int, default=99)
args = p.parse_args()

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})


def calc_entropy(text: str) -> float:
    if not text:
        return 0.0
    return -sum((text.count(c) / len(text)) * math.log2(text.count(c) / len(text))
                for c in set(text))


SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "bank", "account",
    "update", "sign", "confirm", "wallet", "password", "credential",
]
_KW_PAT  = "|".join(SUSPICIOUS_KEYWORDS)
_SPECIAL = r"[@?&=_%;]"
_IP      = r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}"
MAX_TFIDF = 100

HAND_NAMES = [
    "url_length", "num_dots", "num_slashes", "num_digits",
    "num_hyphens", "num_special", "has_https", "has_ip",
    "suspicious_words", "digit_ratio", "special_ratio", "entropy",
]
PRETTY = {
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


def extract_features(urls: pd.Series, tfidf) -> tuple[np.ndarray, list[str]]:
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
    X = pd.concat([lex, tfidf_df], axis=1).to_numpy(dtype=float)
    return X, list(lex.columns)


# ── 1. Load data ─────────────────────────────────────────────────────────── #
print("="*60)
print("  ShieldPhish — SHAP Summary Plot  (XGBoost)")
print("="*60)

print(f"\n[1/3] Loading {args.n_load:,} URLs …")
df = pd.read_csv(DATA_CSV).dropna(subset=["url"])
df["url"]   = df["url"].astype(str)
df["label"] = df["label"].astype(int)
df = df.sample(n=min(args.n_load, len(df)), random_state=args.seed)
print(f"  Legit: {(df['label']==0).sum():,}  |  Phishing: {(df['label']==1).sum():,}")

# ── 2. TF-IDF ────────────────────────────────────────────────────────────── #
tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
with open(tfidf_path, "rb") as f:
    tfidf = pickle.load(f)
if not hasattr(tfidf, "idf_"):
    print("  ⚠ TF-IDF mismatch — re-fitting …")
    df_fit = pd.read_csv(DATA_CSV).dropna(subset=["url"])
    tfidf  = TfidfVectorizer(max_features=MAX_TFIDF, analyzer="char", ngram_range=(3,5))
    tfidf.fit(df_fit["url"].sample(200_000, random_state=42).tolist())

print("[2/3] Extracting features …")
X_arr, feat_names = extract_features(df["url"], tfidf)
y = df["label"].to_numpy()
print(f"  Feature matrix: {X_arr.shape[0]:,} × {X_arr.shape[1]}")

# ── 3. Load XGBoost & compute SHAP ───────────────────────────────────────── #
print(f"[3/3] Loading XGBoost and computing SHAP on {args.n_shap} samples …")
xgb_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
with open(xgb_path, "rb") as f:
    xgb_model = pickle.load(f)

try:
    import shap
except ImportError:
    sys.exit("❌  shap not installed. Run: pip install shap")

# Stratified SHAP sample: equal legit/phishing for balanced explanation
np.random.seed(42)
idx_phish = np.where(y == 1)[0]
idx_legit = np.where(y == 0)[0]
n_each    = min(args.n_shap // 2, len(idx_phish), len(idx_legit))
shap_idx  = np.concatenate([
    np.random.choice(idx_phish, n_each, replace=False),
    np.random.choice(idx_legit, n_each, replace=False),
])
np.random.shuffle(shap_idx)
X_shap = X_arr[shap_idx]
y_shap = y[shap_idx]
print(f"  SHAP sample: {len(shap_idx)} URLs  "
      f"({n_each} phishing + {n_each} legit)")

explainer = shap.TreeExplainer(xgb_model)
shap_vals = explainer.shap_values(X_shap)   # XGBoost returns single array
if isinstance(shap_vals, list):
    sv = shap_vals[1]
else:
    sv = shap_vals

print("  ✔  SHAP values computed")

# ── Aggregate TF-IDF → 1 column for readability ──────────────────────────── #
hc_n      = len(feat_names)          # 12 handcrafted features
sv_hand   = sv[:, :hc_n]
sv_tfidf  = sv[:, hc_n:]
tfidf_agg = sv_tfidf.sum(axis=1, keepdims=True)   # aggregate TF-IDF contribution

X_hand    = X_shap[:, :hc_n]
X_tfidf_m = X_shap[:, hc_n:].mean(axis=1, keepdims=True)

sv_plot   = np.hstack([sv_hand, tfidf_agg])
X_plot    = np.hstack([X_hand,  X_tfidf_m])
plot_names = feat_names + ["char TF-IDF (avg)"]
pretty    = [PRETTY.get(n, n) for n in plot_names]

# Sort by mean |SHAP|, show top 13
mean_abs  = np.abs(sv_plot).mean(axis=0)
order     = np.argsort(mean_abs)          # ascending → bottom of chart = highest
sv_sorted = sv_plot[:, order]
X_sorted  = X_plot[:,  order]
names_sorted = [pretty[i] for i in order]

# ── Draw beeswarm ─────────────────────────────────────────────────────────── #
cmap     = plt.cm.RdBu_r          # red = high feature value, blue = low
n_feats  = sv_sorted.shape[1]

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")

for fi in range(n_feats):
    sv_col  = sv_sorted[:, fi]
    xv_col  = X_sorted[:,  fi]
    y_pos   = fi                  # lowest mean |SHAP| at bottom

    # Normalise feature value for colour
    vmin, vmax = xv_col.min(), xv_col.max()
    norm_xv    = (xv_col - vmin) / (vmax - vmin + 1e-9)

    # Jitter vertically to avoid overplotting
    jitter = np.random.uniform(-0.3, 0.3, size=len(sv_col))
    colors = cmap(norm_xv)
    ax.scatter(sv_col, y_pos + jitter,
               c=colors, s=12, alpha=0.7, linewidths=0, zorder=3)

# Reference line
ax.axvline(0, color="#666666", linewidth=1.1, zorder=1)

# Colour bar
sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.55, aspect=18)
cbar.set_label("Feature value", fontsize=9)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["Low", "High"], fontsize=8)

# Axis labels & formatting
ax.set_yticks(range(n_feats))
ax.set_yticklabels(names_sorted, fontsize=10.5)
ax.set_xlabel("SHAP Value  (impact on phishing probability →)",
              fontsize=11)
ax.set_title("SHAP Feature Impact — XGBoost  (ShieldPhish)",
             fontsize=14, fontweight="bold", pad=14)
ax.tick_params(axis="x", labelsize=9)

# Annotation arrows at the bottom
ax.annotate("← Pushes toward\n    Legitimate",
            xy=(ax.get_xlim()[0]*0.85, -0.7), fontsize=8,
            color="#2563EB", ha="center", style="italic")
ax.annotate("Pushes toward\nPhishing →",
            xy=(ax.get_xlim()[1]*0.85, -0.7), fontsize=8,
            color="#DC2626", ha="center", style="italic")

fig.tight_layout()

for ext in ("png", "pdf"):
    out = os.path.join(OUT_DIR, f"shap_summary.{ext}")
    fig.savefig(out, bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"\n  ✔  Saved → shap_summary.png / .pdf")
print(f"  Output dir: {OUT_DIR}")
print("\nDone ✓")
