import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from scipy import stats

# Allow importing src/ from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.logistic_scratch import LogisticRegressionScratch

def main():
    print("Loading features...")
    df = pd.read_csv("data/features.csv")
    
    # Extract the exact 49,999 evaluation subset described in the paper
    print("Extracting stratified 49,999-URL subset...")
    df_legit = df[df["label"] == 0].sample(n=38577, random_state=42)
    df_phish = df[df["label"] == 1].sample(n=11422, random_state=42)
    df_subset = pd.concat([df_legit, df_phish]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = df_subset.drop(["url", "label"], axis=1)
    y = df_subset["label"]
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    rf_accs = []
    xgb_accs = []
    svm_accs = []
    lr_accs = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\n--- Fold {fold+1} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # 1. Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42, class_weight="balanced")
        rf.fit(X_train, y_train)
        acc_rf = rf.score(X_val, y_val)
        rf_accs.append(acc_rf)
        print(f"RF Acc: {acc_rf:.4f}")
        
        # 2. XGBoost
        print("Training XGBoost...")
        _neg, _pos = (y_train == 0).sum(), (y_train == 1).sum()
        _spw = _neg / _pos
        xgb_model = xgb.XGBClassifier(n_estimators=300, eval_metric="logloss", n_jobs=-1, random_state=42, scale_pos_weight=_spw)
        xgb_model.fit(X_train, y_train)
        acc_xgb = xgb_model.score(X_val, y_val)
        xgb_accs.append(acc_xgb)
        print(f"XGB Acc: {acc_xgb:.4f}")
        
        # 3. SVM
        print("Training Calibrated SVM...")
        svm_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(dual=False, max_iter=2000, random_state=42, class_weight="balanced"),
                cv=3
            )),
        ])
        svm_pipe.fit(X_train, y_train)
        acc_svm = svm_pipe.score(X_val, y_val)
        svm_accs.append(acc_svm)
        print(f"SVM Acc: {acc_svm:.4f}")
        
        # 4. Logistic Regression Scratch
        print("Training Logistic Regression Scratch...")
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        lr = LogisticRegressionScratch(learning_rate=0.1, num_iterations=200)
        lr.fit(X_train_sc, y_train.to_numpy())
        acc_lr = lr.score(X_val_sc, y_val.to_numpy())
        lr_accs.append(acc_lr)
        print(f"LR Acc: {acc_lr:.4f}")

    print("\n" + "="*50)
    print("CROSS VALIDATION ACCURACY SCORES")
    print("="*50)
    print(f"RF Accuracies : {rf_accs} | Mean = {np.mean(rf_accs):.4f} ± {np.std(rf_accs):.4f}")
    print(f"XGB Accuracies: {xgb_accs} | Mean = {np.mean(xgb_accs):.4f} ± {np.std(xgb_accs):.4f}")
    print(f"SVM Accuracies: {svm_accs} | Mean = {np.mean(svm_accs):.4f} ± {np.std(svm_accs):.4f}")
    print(f"LR Accuracies : {lr_accs}  | Mean = {np.mean(lr_accs):.4f} ± {np.std(lr_accs):.4f}")

    # Paired t-tests
    print("\n" + "="*50)
    print("PAIRED T-TEST RESULTS (confidence level 95%, alpha=0.05)")
    print("="*50)
    
    comparisons = [
        ("RF vs XGB", rf_accs, xgb_accs),
        ("RF vs SVM", rf_accs, svm_accs),
        ("RF vs LR", rf_accs, lr_accs)
    ]
    
    results = []
    for label, model_a, model_b in comparisons:
        t_stat, p_val = stats.ttest_rel(model_a, model_b)
        is_sig = "Yes" if p_val < 0.05 else "No"
        p_str = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
        results.append({
            "Comparison": label,
            "Metric": "Accuracy",
            "p-value": p_str,
            "Significant": is_sig,
            "t-statistic": t_stat
        })
        print(f"{label:<10} | t-stat = {t_stat:+7.3f} | p-value = {p_str:<6} | Significant = {is_sig}")

    summary_df = pd.DataFrame(results)
    summary_df.to_csv("graphs/ieee_significance_tests.csv", index=False)
    print("\nSaved significance metrics to graphs/ieee_significance_tests.csv")

if __name__ == "__main__":
    main()
