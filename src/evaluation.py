"""
src/evaluation.py
-----------------
Evaluation utilities: metrics computation, plotting (confusion matrix,
ROC curve, bar charts, rule-weight sensitivity), and CSV export.

All plot functions save both PNG and SVG (and PDF where noted) for IEEE
publication-quality vector graphics.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")           # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ──────────────────────────────────────────────────────────────────────────── #
# Metrics                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

def evaluate_model(
    y_true,
    y_pred,
    model_name: str,
    y_prob=None,
) -> dict:
    """
    Compute classification metrics for a single model.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_pred : array-like
        Predicted binary labels.
    model_name : str
        Label used in result dict and plots.
    y_prob : array-like, optional
        Probability scores for the positive class (enables AUC).

    Returns
    -------
    dict
        Keys: Model, Accuracy, Precision, Recall, F1-Score, AUC.
    """
    metrics = {
        "Model":     model_name,
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
        "AUC":       None,
    }
    if y_prob is not None:
        try:
            metrics["AUC"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            pass
    return metrics


# ──────────────────────────────────────────────────────────────────────────── #
# Plotting helpers                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def _savefig(fig: plt.Figure, base_path: str, formats=("png", "svg")) -> None:
    """Save figure to multiple formats and close it."""
    for ext in formats:
        fig.savefig(f"{base_path}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_metrics(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Bar-chart comparison of Accuracy, Precision, Recall, F1-Score, and AUC
    across all evaluated models.
    """
    os.makedirs(output_dir, exist_ok=True)
    for metric in ("Accuracy", "Precision", "Recall", "F1-Score", "AUC"):
        if metric not in results_df.columns or results_df[metric].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=results_df, x="Model", y=metric, palette="viridis", ax=ax)
        ax.set_title(f"Model Comparison — {metric}", fontsize=14)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=15)
        base = os.path.join(output_dir, f"{metric.lower().replace('-', '_')}_comparison")
        _savefig(fig, base)


def plot_confusion_matrix(y_true, y_pred, model_name: str, output_dir: str) -> None:
    """Heatmap confusion matrix saved as PNG + SVG."""
    os.makedirs(output_dir, exist_ok=True)
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    safe_name = model_name.replace(" ", "_").lower()
    base = os.path.join(output_dir, f"cm_{safe_name}")
    _savefig(fig, base)


def plot_roc_curve(
    y_true,
    y_prob,
    model_name: str,
    output_dir: str,
    suffix: str | None = None,
) -> None:
    """Single-model ROC curve saved as PNG + SVG + PDF."""
    os.makedirs(output_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val     = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"{model_name}  (AUC = {auc_val:.3f})", color="darkorange", lw=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13)
    ax.legend(loc="lower right")

    safe_name = model_name.replace(" ", "_").lower()
    base = os.path.join(output_dir, f"roc_{safe_name}")
    if suffix:
        base += f"_{suffix}"
    _savefig(fig, base, formats=("png", "svg", "pdf"))


def plot_mean_roc_curve(
    mean_fpr: np.ndarray,
    mean_tpr: np.ndarray,
    mean_auc: float,
    model_name: str,
    output_dir: str,
) -> None:
    """Mean ROC curve across CV folds, saved as PNG + SVG + PDF."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mean_fpr, mean_tpr, label=f"{model_name}  Mean AUC = {mean_auc:.3f}", color="navy", lw=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Mean ROC Curve — {model_name} (CV)", fontsize=13)
    ax.legend(loc="lower right")

    safe_name = model_name.replace(" ", "_").lower()
    base = os.path.join(output_dir, f"roc_{safe_name}_mean")
    _savefig(fig, base, formats=("png", "svg", "pdf"))


def plot_rule_weight_sweep(
    weights: list[float],
    false_positives: list[int],
    false_negatives: list[int],
    output_dir: str,
) -> None:
    """
    Plot FP / FN counts as a function of rule_weight.

    Used to justify the chosen rule_weight value in the IEEE paper (§IV-D).
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(weights, false_positives,  marker="o", color="red",  label="False Positives")
    ax.plot(weights, false_negatives,  marker="o", color="blue", label="False Negatives")
    ax.set_xlabel("Rule Weight (W_rule)")
    ax.set_ylabel("Count")
    ax.set_title("Rule Weight Sensitivity — FP and FN vs W_rule", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.4)
    base = os.path.join(output_dir, "rule_weight_sensitivity")
    _savefig(fig, base, formats=("png", "svg", "pdf"))


# ──────────────────────────────────────────────────────────────────────────── #
# CSV export                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

def save_metrics_to_csv(
    results: list[dict],
    output_dir: str,
    filename: str = "metrics_summary.csv",
) -> pd.DataFrame:
    """Persist a list of metric dicts to CSV and return as DataFrame."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"Metrics saved → {os.path.join(output_dir, filename)}")
    return df


if __name__ == "__main__":
    print("Evaluation module loaded. Import and call individual functions.")
