import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.preprocessing import clean_data
from src.feature_engineering import FeatureExtractor

print("Loading data...")
df = pd.read_csv("data/phishing_site_urls.csv")
df = df.rename(columns={"URL": "url", "Label": "label"})
df.columns = [c.lower() for c in df.columns]

# map 'bad' -> 1, 'good' -> 0
if df["label"].dtype == object:
    df["label"] = df["label"].map({"bad": 1, "good": 0})

df = clean_data(df)
df = df.sample(n=50000, random_state=42)

print("Extracting features...")
extractor = FeatureExtractor(use_tfidf=True, max_tfidf_features=100)
extractor.load_vectorizer("models/tfidf_vectorizer.pkl")

features_df = extractor.extract_features(df)
features_df = features_df.dropna()

X = features_df.drop(columns=["url", "label"])
y = features_df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "XGBoost": "models/xgb_model.pkl",
    "Random Forest": "models/rf_model.pkl",
    "Calibrated Linear SVM": "models/svm_model.pkl"
}

results = []
for name, path in models.items():
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        acc = accuracy_score(y_test, y_pred)
        p = precision_score(y_test, y_pred)
        r = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results.append(f"| **{name}** | {acc*100:.2f}% | {p*100:.2f}% | {r*100:.2f}% | {f1*100:.2f}% | {auc:.4f} |")
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

print("\n".join(results))
