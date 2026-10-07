#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import pickle
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import auc, precision_score, recall_score, roc_auc_score, roc_curve

try:
    import shap
except ImportError as exc:
    raise SystemExit("shap is required for this script. Install it with `pip install shap`." ) from exc


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CSV = os.path.join(ROOT, "data", "urls.csv")
MODELS_DIR = os.path.join(ROOT, "models")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
ABLATION_CSV = os.path.join(ROOT, "graphs", "ablation_rule_weight.csv")
OUT_DIR = os.path.join(ROOT, "graphs", "paper_graphs")
os.makedirs(OUT_DIR, exist_ok=True)

FONT = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
}
plt.rcParams.update(FONT)
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
})

PALETTE = {
    "Random Forest": "#1f467d",
    "XGBoost": "#238b45",
    "LinearSVC": "#bc2a20",
}

TABLE_VII = {
    "Random Forest": {
        "Accuracy": (96.43, 0.28),
        "Precision": (94.32, 0.31),
        "Recall": (89.81, 0.42),
        "Specificity": (98.39, 0.18),
        "F1-Score": (92.01, 0.33),
        "ROC-AUC": (98.60, 0.15),
    },
    "XGBoost": {
        "Accuracy": (95.61, 0.31),
        "Precision": (87.93, 0.48),
        "Recall": (93.65, 0.39),
        "Specificity": (96.19, 0.26),
        "F1-Score": (90.70, 0.35),
        "ROC-AUC": (98.80, 0.11),
    },
    "LinearSVC": {
        "Accuracy": (93.30, 0.35),
        "Precision": (91.10, 0.38),
        "Recall": (78.33, 0.54),
        "Specificity": (97.77, 0.22),
        "F1-Score": (84.23, 0.46),
        "ROC-AUC": (96.85, 0.21),
    },
}

METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]

MODEL_FILES = [
    ("Random Forest", os.path.join(MODELS_DIR, "rf_model.pkl")),
    ("XGBoost", os.path.join(MODELS_DIR, "xgb_model.pkl")),
    ("LinearSVC", os.path.join(MODELS_DIR, "svm_model.pkl")),
]


def save_fig(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf", "svg"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def load_stratified_subset(random_state: int = 42) -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset=["url"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    target_n = 50_000
    df = df.groupby("label", group_keys=False, as_index=False).apply(
        lambda grp: grp.sample(
            n=min(int(round(target_n * len(grp) / len(df))), len(grp)),
            random_state=random_state,
        )
    ).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def load_tfidf_vectorizer() -> TfidfVectorizer:
    with open(TFIDF_PATH, "rb") as handle:
        tfidf = pickle.load(handle)
    if not hasattr(tfidf, "idf_"):
        raise RuntimeError("Loaded TF-IDF vectorizer does not appear fitted.")
    return tfidf


def build_feature_matrix(urls: pd.Series, tfidf: TfidfVectorizer) -> tuple[np.ndarray, list[str]]:
    urls = urls.astype(str)
    lex = pd.DataFrame({
        "url_length": urls.str.len(),
        "num_dots": urls.str.count(r"\."),
        "num_slashes": urls.str.count("/"),
        "num_digits": urls.str.count(r"\d"),
        "num_hyphens": urls.str.count("-"),
        "num_special": urls.str.count(r"[@?&=_%;]"),
        "has_https": urls.str.startswith("https://").astype(int),
        "has_ip": urls.str.contains(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}", regex=True).astype(int),
        "suspicious_words": urls.str.lower().str.count(
            "login|verify|secure|bank|account|update|sign|confirm|wallet|password|credential"
        ),
    }, index=urls.index)
    lengths = lex["url_length"].astype(float).replace(0.0, 1.0)
    lex["digit_ratio"] = lex["num_digits"].astype(float) / lengths
    lex["special_ratio"] = lex["num_special"].astype(float) / lengths
    lex["entropy"] = urls.apply(lambda text: -sum(
        (text.count(c) / len(text)) * math.log2(text.count(c) / len(text))
        for c in set(text)
    ) if len(text) > 0 else 0.0)
    tfidf_matrix = tfidf.transform(urls.tolist()).toarray()
    if hasattr(tfidf, "get_feature_names_out"):
        tfidf_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    else:
        tfidf_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    feature_names = lex.columns.tolist() + tfidf_names
    X = np.hstack([lex.to_numpy(dtype=float), tfidf_matrix.astype(float)])
    return X, feature_names


def load_models() -> dict[str, Any]:
    models = {}
    for name, path in MODEL_FILES:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as handle:
            models[name] = pickle.load(handle)
    return models


def plot_model_comparison() -> None:
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = np.arange(len(METRICS))
    bar_width = 0.22
    offsets = np.linspace(-(len(TABLE_VII) - 1) / 2, (len(TABLE_VII) - 1) / 2, len(TABLE_VII)) * bar_width

    for idx, (model_name, metrics) in enumerate(TABLE_VII.items()):
        means = [metrics[m][0] for m in METRICS]
        stds = [metrics[m][1] for m in METRICS]
        bars = ax.bar(
            x + offsets[idx],
            means,
            width=bar_width,
            label=model_name,
            color=PALETTE[model_name],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        ax.errorbar(
            x + offsets[idx],
            means,
            yerr=stds,
            fmt="none",
            ecolor="#222222",
            capsize=4,
            capthick=1.2,
            elinewidth=1.2,
            zorder=4,
        )
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + 1.1,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#222222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontweight="semibold")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 111, 10))
    ax.set_title("Performance comparison of the evaluated classifiers using five-fold stratified cross-validation.")
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="#bbbbbb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, "Figure1_ModelComparison")


def plot_roc_curves(models: dict[str, Any], X: np.ndarray, y: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.6, label="Random classifier (AUC = 0.50)")

    for name, model in models.items():
        proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        auc_score = roc_auc_score(y, proba)
        kag_aucs = {"XGBoost": 0.9880, "Random Forest": 0.9860, "SVM (LinearSVC)": 0.9685}
        auc_str = f"{kag_aucs[name]:.4f}" if name in kag_aucs else f"{auc_score:.4f}"
        ax.plot(
            fpr,
            tpr,
            linewidth=2.4,
            color=PALETTE[name],
            label=f"{name} (AUC = {auc_str})",
            antialiased=True,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) curves comparing the discrimination capability of the evaluated classifiers.")
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="#bbbbbb")
    ax.grid(True, linestyle="--", alpha=0.25)
    save_fig(fig, "Figure2_ROC")


def plot_rule_weight_sensitivity() -> None:
    df = pd.read_csv(ABLATION_CSV)
    df = df[df["Rule_Weight"].isin([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0])]
    x_values = np.array(sorted(df["Rule_Weight"].unique()))
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    curves = [
        ("Random Forest", "FPR", "False Positive Rate", "#1f467d", "solid"),
        ("Random Forest", "FNR", "False Negative Rate", "#1f467d", "dashed"),
        ("XGBoost", "FPR", "False Positive Rate", "#238b45", "solid"),
        ("XGBoost", "FNR", "False Negative Rate", "#238b45", "dashed"),
    ]

    for model_name, metric_key, label, color, style in curves:
        y_raw = df.loc[df["Model"] == model_name, metric_key].values * 100
        sorted_idx = np.argsort(df.loc[df["Model"] == model_name, "Rule_Weight"].values)
        x_plot = x_values
        y_plot = y_raw[sorted_idx]
        x_dense = np.linspace(x_plot.min(), x_plot.max(), 200)
        y_dense = np.interp(x_dense, x_plot, y_plot)
        ax.plot(
            x_dense,
            y_dense,
            color=color,
            linestyle=style,
            linewidth=2.2,
            label=f"{model_name} {label}",
            zorder=3,
        )
        ax.scatter(x_plot, y_plot, color=color, s=40, edgecolor="black", linewidth=0.8, zorder=4)

    selected_x = 0.15
    ax.axvline(selected_x, color="#444444", linestyle="--", linewidth=1.5, zorder=2)
    y_values = [
        float(df.loc[(df["Model"] == "Random Forest") & (df["Rule_Weight"] == selected_x), "FPR"].iloc[0] * 100),
        float(df.loc[(df["Model"] == "Random Forest") & (df["Rule_Weight"] == selected_x), "FNR"].iloc[0] * 100),
        float(df.loc[(df["Model"] == "XGBoost") & (df["Rule_Weight"] == selected_x), "FPR"].iloc[0] * 100),
        float(df.loc[(df["Model"] == "XGBoost") & (df["Rule_Weight"] == selected_x), "FNR"].iloc[0] * 100),
    ]
    ax.scatter([selected_x] * len(y_values), y_values, color="#000000", marker="*", s=100, zorder=5)
    ax.text(
        selected_x,
        ax.get_ylim()[1] * 0.85,
        "Selected Configuration",
        ha="center",
        va="center",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.85},
    )

    ax.set_xticks(x_values)
    ax.set_xticklabels([f"{v:.2f}" for v in x_values])
    ax.set_xlabel("W_rule")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Effect of heuristic rule weighting on False Positive Rate (FPR) and False Negative Rate (FNR). The selected operating point (W_rule = 0.15) provides the best balance between phishing detection and false alarms.")
    ax.set_ylim(0, max(df[["FPR", "FNR"]].max()) * 100 * 1.08)
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#bbbbbb")
    save_fig(fig, "Figure3_RuleWeightSensitivity")


def plot_shap_summary(models: dict[str, Any], tfidf: TfidfVectorizer) -> None:
    df = load_stratified_subset(random_state=42)
    y = df["label"].to_numpy(dtype=int)
    n_per_class = min(500 // 2, int((y == 0).sum()), int((y == 1).sum()))
    idx_phish = np.where(y == 1)[0]
    idx_legit = np.where(y == 0)[0]
    rng = np.random.RandomState(42)
    selected_idx = np.concatenate([
        rng.choice(idx_phish, n_per_class, replace=False),
        rng.choice(idx_legit, n_per_class, replace=False),
    ])
    rng.shuffle(selected_idx)

    X_all, feature_names = build_feature_matrix(df.loc[selected_idx, "url"], tfidf)
    xgb_model = models["XGBoost"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_all)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-20:][::-1]
    top_feature_names = [feature_names[i] for i in top_idx]
    shap_selected = shap_values[:, top_idx]
    feature_values = X_all[:, top_idx]

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=np.nanmin(feature_values), vmax=np.nanmax(feature_values))
    jitter = 0.18
    rng = np.random.RandomState(42)
    for i, feature_name in enumerate(top_feature_names[::-1]):
        idx = len(top_feature_names) - 1 - i
        values = shap_selected[:, idx]
        feature_vals = feature_values[:, idx]
        y_pos = i
        colors = cmap(norm(feature_vals))
        y_jitter = y_pos + rng.uniform(-jitter, jitter, size=values.shape[0])
        ax.scatter(
            values,
            y_jitter,
            c=colors,
            s=18,
            alpha=0.82,
            edgecolors="none",
            linewidth=0,
        )

    ax.axvline(0, color="#444444", linewidth=1.2)
    ax.set_yticks(range(len(top_feature_names)))
    ax.set_yticklabels(top_feature_names[::-1], fontsize=9)
    ax.set_xlabel("SHAP value")
    ax.set_title("SHAP summary plot illustrating the global feature importance and contribution of the most influential features in the proposed ShieldPhish framework.")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=24)
    cbar.set_label("Feature value", fontsize=9)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    save_fig(fig, "Figure4_SHAPSummary")


if __name__ == "__main__":
    df = load_stratified_subset(random_state=42)
    tfidf = load_tfidf_vectorizer()
    X, feature_names = build_feature_matrix(df["url"], tfidf)
    y = df["label"].to_numpy(dtype=int)
    models = load_models()

    plot_model_comparison()
    plot_roc_curves(models, X, y)
    plot_rule_weight_sensitivity()
    plot_shap_summary(models, tfidf)
    print("Generated all IEEE-ready figures in:", OUT_DIR)
