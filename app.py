"""
app.py  (project root)
-----------------------
Streamlit interactive UI for the PhishDetect phishing detection system.

Run
---
    streamlit run app.py
"""

from __future__ import annotations

import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st

from src.feature_engineering import FeatureExtractor
from src.hybrid_model import HybridPhishingModel

# ── Page config ──────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="PhishDetect — URL Scanner",
    page_icon="🛡️",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────── #
st.markdown(
    """
    <style>
    .phishing   { color: #ff4b4b; font-weight: bold; font-size: 24px; }
    .legitimate { color: #00cc66; font-weight: bold; font-size: 24px; }
    .metric-box {
        background: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Model loading (cached) ───────────────────────────────────────────────── #
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


@st.cache_resource
def load_models():
    class _Dummy:
        def predict_proba(self, X):
            return np.array([[0.1, 0.9]])

    model_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

    try:
        with open(model_path, "rb") as f:
            ml_model = pickle.load(f)
    except FileNotFoundError:
        st.warning(
            "⚠️ XGBoost model not found — using dummy model.  "
            "Run `python scripts/train_all.py` to train."
        )
        ml_model = _Dummy()

    ext = FeatureExtractor(use_tfidf=True, max_tfidf_features=100)
    ext.load_vectorizer(tfidf_path)
    hybrid = HybridPhishingModel(ml_model=ml_model, rule_weight=0.15)
    return ext, hybrid


extractor, hybrid_system = load_models()

# ── UI ───────────────────────────────────────────────────────────────────── #
st.title("🛡️ PhishDetect — Real-Time URL Scanner")
st.markdown(
    "**Hybrid Model**: Machine Learning probability fused with "
    "heuristic rule-based scoring for explainable phishing detection."
)

st.subheader("Analyse a URL")
url_input = st.text_input(
    "Enter URL to scan:",
    placeholder="https://www.example.com",
    key="url_input",
)

if st.button("🔍 Detect Phishing", type="primary", key="detect_btn"):
    if not url_input.strip():
        st.error("Please enter a valid URL.")
    else:
        with st.spinner("Analysing URL…"):
            time.sleep(0.6)          # brief UX pause

            df_single   = pd.DataFrame({"url": [url_input]})
            df_features = extractor.extract_features(df_single.copy())
            X_input     = df_features.drop(["url"], axis=1)
            result      = hybrid_system.predict(url_input, X_input)

        st.divider()

        # ── Result header ─── #
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Verdict")
            if result["prediction"] == "Phishing":
                st.markdown(
                    f"<span class='phishing'>🚨 {result['prediction']}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<span class='legitimate'>✅ {result['prediction']}</span>",
                    unsafe_allow_html=True,
                )

        with col2:
            st.markdown("### Risk Score")
            st.progress(result["final_score"])
            st.write(f"**{result['final_score'] * 100:.1f}%** danger rate")

        # ── Score breakdown ─── #
        st.markdown("### Breakdown")
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**ML Probability:** {result['ml_probability']*100:.1f}%")
        with c2:
            st.warning(f"**Rule Penalty Score:** {result['rule_score']} pts")

        # ── Rule engine insights ─── #
        st.markdown("### 🔍 Rule Engine Insights")
        if result["rule_details"]:
            for detail in result["rule_details"]:
                st.write(f"- {detail}")
        else:
            st.success("No suspicious rules triggered.")

        # ── SHAP XAI reasons ─── #
        if result.get("xai_reasons"):
            st.markdown("### 🧠 Top ML Feature Contributions")
            for reason in result["xai_reasons"]:
                st.write(f"- {reason}")

        # ── Raw feature matrix ─── #
        with st.expander("🔬 View Extracted Feature Matrix"):
            st.dataframe(X_input)
