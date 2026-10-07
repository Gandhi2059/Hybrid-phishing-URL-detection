"""
scripts/ablation_rule_weight.py
------------------------------
Run a parameter sensitivity (ablation) sweep on rule_weight (W_rule) from 0.0 to 1.0.
Computes Accuracy, F1-Score, False Positive Rate (FPR), and False Negative Rate (FNR)
for SVM, Random Forest, and XGBoost models.
Saves results to a CSV and generates publication-grade PNG and PDF plots.

Usage
-----
    python scripts/ablation_rule_weight.py --n-samples 20000
"""

import argparse
import os
import pickle
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.rule_engine import RuleEngine

# ── Arguments ─────────────────────────────────────────────────────────────── #
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/features.csv", help="Path to features CSV")
parser.add_argument("--n-samples", type=int, default=0, help="0 = full 50k benchmark; >0 = subsample")
args = parser.parse_args()

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT, "models")
GRAPHS_DIR = os.path.join(ROOT, "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

def main():
    print(f"Loading full features from {args.data}...")
    df_full = pd.read_csv(os.path.join(ROOT, args.data))
    
    # Stratified 50,000 subsample matching the paper's benchmark dataset
    TARGET_N = 50_000
    df = df_full.groupby("label", group_keys=False).apply(
        lambda grp: grp.sample(
            n=min(int(round(TARGET_N * len(grp) / len(df_full))), len(grp)),
            random_state=42
        )
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    if args.n_samples > 0 and args.n_samples < len(df):
        print(f"Subsampling to {args.n_samples:,} rows...")
        df = df.sample(n=args.n_samples, random_state=42).reset_index(drop=True)

    print(f"Benchmark dataset ready: {len(df):,} URLs ({int((df['label']==0).sum()):,} legit, {int((df['label']==1).sum()):,} phishing)")
    
    urls = df["url"].tolist()
    y = df["label"].values
    X = df.drop(["url", "label"], axis=1)

    # Initialize Rule Engine
    rule_engine = RuleEngine()
    print("Evaluating URLs against deterministic rules...")
    rule_scores = np.array([rule_engine.evaluate(url)[0] for url in urls])

    # Weights to sweep
    weights = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    import xgboost as xgb

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models_to_run = ["SVM", "Random Forest", "XGBoost"]
    all_results = []
    
    # Drop globally-fitted TF-IDF features to prevent leakage; we compute them strictly per-fold
    tfidf_cols = [c for c in X.columns if c.startswith("tfidf_")]
    X_base = X.drop(columns=tfidf_cols)
    urls_np = np.array(urls)

    for model_name in models_to_run:
        print(f"Running 5-fold strictly isolated out-of-fold ablation sweep for: {model_name}...")
        oof_ml_probs = np.zeros(len(y), dtype=float)

        for fold, (tr_idx, te_idx) in enumerate(cv.split(X_base, y), 1):
            X_tr_base, X_te_base = X_base.iloc[tr_idx], X_base.iloc[te_idx]
            y_tr = y[tr_idx]

            # 1. Strict Fold-Isolated TF-IDF
            tfidf = TfidfVectorizer(max_features=100, analyzer="char", ngram_range=(3, 5))
            tfidf_tr = tfidf.fit_transform(urls_np[tr_idx]).toarray()
            tfidf_te = tfidf.transform(urls_np[te_idx]).toarray()

            # 2. Combine handcrafted + TF-IDF
            X_tr = np.hstack([X_tr_base.to_numpy(), tfidf_tr])
            X_te = np.hstack([X_te_base.to_numpy(), tfidf_te])

            # 3. Dynamic scale_pos_weight
            neg = (y_tr == 0).sum()
            pos = (y_tr == 1).sum()
            spw = round(neg / pos, 4)

            # 4. Strict Fold-Isolated StandardScaler
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)

            if model_name == "SVM":
                clf = CalibratedClassifierCV(LinearSVC(loss="squared_hinge", dual=False, max_iter=2000, class_weight="balanced", random_state=42), cv=3)
                clf.fit(X_tr_scaled, y_tr)
                oof_ml_probs[te_idx] = clf.predict_proba(X_te_scaled)[:, 1]
            elif model_name == "Random Forest":
                clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
                clf.fit(X_tr, y_tr)
                oof_ml_probs[te_idx] = clf.predict_proba(X_te)[:, 1]
            elif model_name == "XGBoost":
                clf = xgb.XGBClassifier(n_estimators=300, eval_metric="logloss", scale_pos_weight=spw, random_state=42, n_jobs=-1, verbosity=0)
                clf.fit(X_tr, y_tr)
                oof_ml_probs[te_idx] = clf.predict_proba(X_te)[:, 1]

        for w in weights:
            hybrid_probs = np.minimum(oof_ml_probs + rule_scores * w, 1.0)
            preds = (hybrid_probs >= 0.5).astype(int)

            acc = accuracy_score(y, preds)
            f1 = f1_score(y, preds, zero_division=0)
            cm = confusion_matrix(y, preds)
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            all_results.append({
                "Model": model_name,
                "Rule_Weight": w,
                "Accuracy": acc,
                "F1_Score": f1,
                "FPR": fpr,
                "FNR": fnr,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn
            })

    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(GRAPHS_DIR, "ablation_rule_weight.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"✔ Saved out-of-fold ablation metrics to {csv_path}")

    plot_ablation_results(results_df)

def plot_ablation_results(df):
    models = df["Model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharex=True)
    
    if len(models) == 1:
        axes = [axes]

    for idx, model_name in enumerate(models):
        ax1 = axes[idx]
        model_df = df[df["Model"] == model_name].sort_values("Rule_Weight")
        
        weights = model_df["Rule_Weight"].values
        acc = model_df["Accuracy"].values
        f1 = model_df["F1_Score"].values
        fpr = model_df["FPR"].values
        fnr = model_df["FNR"].values

        # Left Y-axis (Accuracy & F1-Score)
        color = "tab:blue"
        ax1.set_xlabel("Rule Weight ($W_{rule}$)", fontsize=11)
        ax1.set_ylabel("Classification Performance (Accuracy/F1)", color=color, fontsize=11)
        line1 = ax1.plot(weights, acc, label="Accuracy", color=color, linestyle="--", marker="o")
        line2 = ax1.plot(weights, f1, label="F1-Score", color=color, linestyle="-", marker="s")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Right Y-axis (FPR & FNR)
        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Error Rates (FPR/FNR)", color=color, fontsize=11)
        line3 = ax2.plot(weights, fpr, label="FPR (False Alarm)", color=color, linestyle="-", marker="x")
        line4 = ax2.plot(weights, fnr, label="FNR (Miss Rate)", color=color, linestyle="--", marker="^")
        ax2.tick_params(axis="y", labelcolor=color)

        # Combine legends
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="lower center" if idx == 0 else "best")

        ax1.set_title(f"Ablation Sweep: {model_name}", fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    
    # Save as PNG
    png_path = os.path.join(GRAPHS_DIR, "ablation_rule_weight_plot.png")
    plt.savefig(png_path, dpi=300)
    print(f"✔ Saved plot to {png_path}")

    # Save as PDF (Vector graphic for paper submission)
    pdf_path = os.path.join(GRAPHS_DIR, "ablation_rule_weight_plot.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"✔ Saved vector plot to {pdf_path}")
    
    plt.close()

if __name__ == "__main__":
    main()
