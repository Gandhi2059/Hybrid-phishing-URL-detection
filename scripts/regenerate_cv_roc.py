#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import auc, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CSV = os.path.join(ROOT, "data", "urls.csv")
TFIDF_PATH = os.path.join(ROOT, "models", "tfidf_vectorizer.pkl")
OUT_DIR = os.path.join(ROOT, "graphs", "paper_graphs")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
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


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset=["url"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    target_n = 50_000
    df = df.groupby("label", group_keys=False, as_index=False).apply(
        lambda grp: grp.sample(
            n=min(int(round(target_n * len(grp) / len(df))), len(grp)),
            random_state=42,
        )
    ).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def load_tfidf() -> TfidfVectorizer:
    with open(TFIDF_PATH, "rb") as handle:
        tfidf = pd.read_pickle(handle)
    if not hasattr(tfidf, "idf_"):
        raise RuntimeError("Loaded TF-IDF vectorizer is not fitted.")
    return tfidf


def build_feature_dataframe(urls: pd.Series, tfidf: TfidfVectorizer) -> pd.DataFrame:
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
    lengths = lex["url_length"].replace(0, 1).astype(float)
    lex["digit_ratio"] = lex["num_digits"].astype(float) / lengths
    lex["special_ratio"] = lex["num_special"].astype(float) / lengths
    lex["entropy"] = urls.apply(
        lambda text: -sum(
            (text.count(c) / len(text)) * math.log2(text.count(c) / len(text))
            for c in set(text)
        ) if len(text) > 0 else 0.0
    )
    tfidf_matrix = tfidf.transform(urls.tolist()).toarray()
    tfidf_cols = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_cols, index=urls.index)
    return pd.concat([lex, tfidf_df], axis=1)


def build_models() -> dict[str, Any]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, eval_metric="logloss",
            scale_pos_weight=None, n_jobs=-1, random_state=42, verbosity=0,
        ),
        "LinearSVC": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(dual=False, max_iter=2000, random_state=42, class_weight="balanced"),
                cv=3,
            )),
        ]),
    }


def regenerate_cv_predictions(X: pd.DataFrame, y: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = build_models()
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    spw = round(neg / pos, 4)
    models["XGBoost"].set_params(scale_pos_weight=spw)

    aggregated = {}
    for name, model in models.items():
        preds = np.zeros_like(y, dtype=float)
        trues = np.zeros_like(y, dtype=int)
        probs = np.zeros_like(y, dtype=float)

        for _, test_idx in cv.split(X, y):
            train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
            model.fit(X.iloc[train_idx], y[train_idx])
            probs[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
            preds[test_idx] = model.predict(X.iloc[test_idx])
            trues[test_idx] = y[test_idx]

        aggregated[name] = {
            "y_true": trues,
            "y_pred": preds,
            "y_prob": probs,
        }
    return aggregated


def plot_cv_roc(curves: dict[str, dict[str, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.6,
            label="Random classifier (AUC = 0.500)")

    for name, data in curves.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        auc_score = roc_auc_score(data["y_true"], data["y_prob"])
        ax.plot(
            fpr,
            tpr,
            linewidth=2.5,
            color=PALETTE[name],
            label=f"{name} (AUC = {auc_score:.4f})",
            antialiased=True,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        "Receiver Operating Characteristic (ROC) curves from stratified 5-fold cross-validation using the same dataset and protocol as Table VII."
    )
    ax.legend(loc="lower right", framealpha=0.92, edgecolor="#bbbbbb")
    save_paths = [
        os.path.join(OUT_DIR, "Figure2_ROC.png"),
        os.path.join(OUT_DIR, "Figure2_ROC.pdf"),
        os.path.join(OUT_DIR, "Figure2_ROC.svg"),
    ]
    for path in save_paths:
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved ROC plots:", ", ".join(save_paths))


if __name__ == "__main__":
    df = load_dataset()
    X = build_feature_dataframe(df["url"], load_tfidf())
    y = df["label"].to_numpy(dtype=int)
    results = regenerate_cv_predictions(X, y)
    plot_cv_roc(results)
    print("Regenerated ROC curves using exact Table VII cross-validation predictions.")
