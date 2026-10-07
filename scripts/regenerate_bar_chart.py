"""
scripts/regenerate_bar_chart.py
================================
Generates the publication-quality Model Performance Comparison bar chart
strictly conforming to IEEE formatting standards and exactly matching the
numerical values reported in the manuscript (Table VII).

IEEE Design Specifications:
  1. Exact data match with Manuscript Table VII (RF, XGBoost, LinearSVC).
  2. 0–100% Y-axis scale (untruncated, unbiased baseline).
  3. Crisp error bars (±1 std across 5 folds).
  4. De-cluttered: no redundant numeric labels above bars (table contains exact values).
  5. Clean, concise title: "Performance Comparison of Evaluated Classifiers".
  6. Scaled typography for high legibility at IEEE single/double column widths (~8.5–17 cm).
  7. No report-style subheadings or metadata annotations on the canvas.

Usage
-----
    python scripts/regenerate_bar_chart.py
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "graphs", "paper_graphs"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Exact Data from Manuscript Table VII (5-Fold Stratified CV) ───────────── #
# Format: (mean_percentage, std_percentage)
DATA = {
    "Random Forest": {
        "Accuracy":    (95.01, 0.28),
        "Precision":   (93.57, 0.31),
        "Recall":      (83.91, 0.42),
        "Specificity": (98.29, 0.18),
        "F1-Score":    (88.48, 0.33),
        "ROC-AUC":     (98.08, 0.15),
    },
    "XGBoost": {
        "Accuracy":    (94.37, 0.31),
        "Precision":   (85.57, 0.48),
        "Recall":      (90.62, 0.39),
        "Specificity": (95.48, 0.26),
        "F1-Score":    (88.03, 0.35),
        "ROC-AUC":     (98.32, 0.11),
    },
    "SVM (LinearSVC)": {
        "Accuracy":    (93.14, 0.35),
        "Precision":   (91.03, 0.38),
        "Recall":      (77.63, 0.54),
        "Specificity": (97.73, 0.22),
        "F1-Score":    (83.80, 0.46),
        "ROC-AUC":     (96.80, 0.21),
    },
}

METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]
MODELS  = list(DATA.keys())

# IEEE-compliant distinctive color palette (Blue, Green, Red)
PALETTE = {
    "Random Forest":   "#2563EB",   # Royal Blue
    "XGBoost":         "#16A34A",   # Emerald Green
    "SVM (LinearSVC)": "#DC2626",   # Crimson Red
}

# Matplotlib IEEE styling
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "figure.autolayout": False,
})

n_metrics = len(METRICS)
n_models  = len(MODELS)
x         = np.arange(n_metrics)
bar_width = 0.24
offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width

fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FFFFFF")

for i, model_name in enumerate(MODELS):
    color = PALETTE[model_name]
    means = [DATA[model_name][m][0] / 100 for m in METRICS]
    stds  = [DATA[model_name][m][1] / 100 for m in METRICS]

    # Render bars
    ax.bar(
        x + offsets[i], means,
        width=bar_width,
        label=model_name,
        color=color,
        alpha=0.90,
        edgecolor="#1E293B",
        linewidth=0.8,
        zorder=3,
    )

    # Render error bars (±1 std)
    ax.errorbar(
        x + offsets[i], means,
        yerr=stds,
        fmt="none",
        color="#0F172A",
        capsize=4.0,
        capthick=1.3,
        elinewidth=1.3,
        zorder=4,
    )

# ── Axis Formatting & Labels ──────────────────────────────────────────────── #
ax.set_xticks(x)
ax.set_xticklabels(METRICS, fontsize=12.5, fontweight="bold", color="#1E293B")
ax.set_ylabel("Score (%)", fontsize=13, fontweight="bold", color="#1E293B")

# Unbiased 0–100% Y-axis
ax.set_ylim(0.0, 1.05)
ax.set_yticks(np.linspace(0.0, 1.0, 6))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v*100)}%"))
ax.tick_params(axis="y", labelsize=11.5)
ax.tick_params(axis="x", labelsize=12)

# Concise IEEE title
ax.set_title(
    "Performance Comparison of Evaluated Classifiers",
    fontsize=14.5, fontweight="bold", pad=16, color="#0F172A"
)

# Clean, uncluttered legend located in lower right (empty quadrant in 0–100% scale)
ax.legend(
    loc="lower right",
    fontsize=11,
    frameon=True,
    facecolor="#FFFFFF",
    edgecolor="#CBD5E1",
    framealpha=0.95,
    title="Classifier",
    title_fontsize=11.5,
)

fig.tight_layout()

# ── Export high-res PNG, vector PDF, and SVG ──────────────────────────────── #
for ext in ("png", "pdf", "svg"):
    path = os.path.join(OUT_DIR, f"model_comparison_bar.{ext}")
    fig.savefig(path, bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"✔ Successfully generated IEEE publication-ready bar chart:")
print(f"  → {OUT_DIR}/model_comparison_bar.png (300 DPI)")
print(f"  → {OUT_DIR}/model_comparison_bar.pdf (Vector)")
print(f"  → {OUT_DIR}/model_comparison_bar.svg (Vector)")
