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
parser.add_argument("--n-samples", type=int, default=20000, help="Number of sample URLs to evaluate")
args = parser.parse_args()

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT, "models")
GRAPHS_DIR = os.path.join(ROOT, "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

def main():
    print(f"Loading first {args.n_samples:,} rows from {args.data}...")
    df = pd.read_csv(os.path.join(ROOT, args.data), nrows=args.n_samples)
    
    # We already have at most args.n_samples rows
    df = df.reset_index(drop=True)
    
    urls = df["url"].tolist()
    y = df["label"].values
    X = df.drop(["url", "label"], axis=1)

    # Initialize Rule Engine
    rule_engine = RuleEngine()
    print("Evaluating URLs against deterministic rules...")
    # Pre-calculate rule scores for all sample URLs
    rule_scores = np.array([rule_engine.evaluate(url)[0] for url in urls])

    # Weights to sweep
    weights = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    
    MODELS = [
        ("SVM", os.path.join(MODELS_DIR, "svm_model.pkl")),
        ("Random Forest", os.path.join(MODELS_DIR, "rf_model.pkl")),
        ("XGBoost", os.path.join(MODELS_DIR, "xgb_model.pkl")),
    ]

    all_results = []

    for model_name, model_path in MODELS:
        if not os.path.exists(model_path):
            print(f"  [{model_name}] model not found at {model_path}, skipping.")
            continue
        
        print(f"Running ablation sweep for model: {model_name}...")
        with open(model_path, "rb") as f:
            payload = pickle.load(f)

        # Handle data scaling for SVM
        if model_name == "SVM":
            if isinstance(payload, dict):
                model = payload["model"]
                scaler = payload["scaler"]
                features_to_use = getattr(scaler, "feature_names_in_", getattr(model, "feature_names_in_", None))
                X_eval = X[list(features_to_use)] if features_to_use is not None else X
                X_eval = scaler.transform(X_eval)
            else:
                model = payload
                features_to_use = getattr(model, "feature_names_in_", None)
                X_eval = X[list(features_to_use)] if features_to_use is not None else X
        else:
            model = payload
            features_to_use = getattr(model, "feature_names_in_", None)
            X_eval = X[list(features_to_use)] if features_to_use is not None else X

        # Get batch predictions probability from ML model
        # predict_proba returns [prob_legitimate, prob_phishing]
        ml_probs = model.predict_proba(X_eval)[:, 1]

        for w in weights:
            # Bounded addition fusion formula: hybrid_prob = min(ml_prob + rule_score * rule_weight, 1.0)
            hybrid_probs = np.minimum(ml_probs + rule_scores * w, 1.0)
            
            # Prediction decision boundary
            preds = (hybrid_probs >= 0.5).astype(int)

            # Compute metrics
            acc = accuracy_score(y, preds)
            f1 = f1_score(y, preds)
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

    # Save to CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(GRAPHS_DIR, "ablation_rule_weight.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"✔ Saved raw metrics to {csv_path}")

    # Plot metrics
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
