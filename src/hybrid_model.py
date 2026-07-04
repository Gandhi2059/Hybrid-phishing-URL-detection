"""
src/hybrid_model.py
--------------------
Hybrid phishing detection: ML probability + heuristic rule score + SHAP XAI.

Fusion formula (bounded addition)
----------------------------------
    hybrid_prob = min(ml_prob + rule_score × rule_weight, 1.0)

where rule_weight = 0.15 by default, balancing ML confidence with
lightweight heuristic evidence without over-penalising low-risk URLs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rule_engine import RuleEngine


class HybridPhishingModel:
    """
    Combine an ML classifier with the heuristic RuleEngine.

    Parameters
    ----------
    ml_model : sklearn-compatible estimator
        Must implement ``predict_proba(X)``.
    rule_weight : float
        Scales the rule score's contribution to the final probability.
        Range: 0.0 (rules ignored) – 1.0 (full rule influence).
        Default: 0.15 (chosen empirically; see details.md §4).
    """

    def __init__(self, ml_model, rule_weight: float = 0.15):
        self.ml_model    = ml_model
        self.rule_engine = RuleEngine()
        self.rule_weight = rule_weight

        # SHAP explainer — optional; gracefully degrades if unavailable
        self.explainer = None
        try:
            import shap
            self.explainer = shap.TreeExplainer(ml_model)
        except Exception:
            pass  # non-tree models or missing shap — XAI silently disabled

    def predict(
        self,
        url: str,
        features_df: pd.DataFrame,
        threshold: float = 0.5,
        explain: bool = True,
    ) -> dict:
        """
        Predict whether a single URL is phishing or legitimate.

        Parameters
        ----------
        url : str
            Raw URL string (passed to the rule engine).
        features_df : pd.DataFrame
            Pre-computed feature row (from FeatureExtractor.extract_features).
        threshold : float
            Decision boundary for the final hybrid probability.
        explain : bool
            Generate SHAP-based XAI reasons when True and an explainer is
            available.

        Returns
        -------
        dict with keys:
            prediction    (str)   — "Phishing" | "Legitimate"
            final_score   (float) — Hybrid probability [0, 1]
            ml_probability (float)
            rule_score    (float)
            rule_details  (list[str])
            xai_reasons   (list[str])
        """
        # 1. ML probability (class-1 / phishing)
        ml_prob = float(self.ml_model.predict_proba(features_df)[0][1])

        # 2. Rule-based score
        rule_score, rule_details = self.rule_engine.evaluate(url)

        # 3. Fusion — clamped to [0, 1]
        hybrid_prob = min(ml_prob + rule_score * self.rule_weight, 1.0)

        # 4. Classification
        prediction = "Phishing" if hybrid_prob >= threshold else "Legitimate"

        # 5. SHAP feature attribution (XAI)
        top_reasons: list[str] = []
        if explain and self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(features_df)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]  # class-1 values
                vals          = shap_vals[0]
                feature_names = features_df.columns
                top_idx       = np.argsort(vals)[::-1]
                for idx in top_idx[:5]:
                    if vals[idx] > 0:
                        feat = feature_names[idx].replace("_", " ").title()
                        top_reasons.append(f"{feat} (SHAP: +{vals[idx]:.2f})")
            except Exception:
                pass

        return {
            "prediction":     prediction,
            "final_score":    round(hybrid_prob, 4),
            "ml_probability": round(ml_prob, 4),
            "rule_score":     float(rule_score),
            "rule_details":   rule_details,
            "xai_reasons":    top_reasons,
        }

    def batch_predict(
        self,
        urls: list[str],
        features_df: pd.DataFrame,
        threshold: float = 0.5,
        explain: bool = False,
    ) -> list[dict]:
        """
        Run predictions on multiple URLs.

        Parameters
        ----------
        urls : list[str]
            Raw URL strings (same order as features_df rows).
        features_df : pd.DataFrame
            One row per URL.

        Returns
        -------
        list[dict]
            One result dict per URL.
        """
        results = []
        for i, url in enumerate(urls):
            row_df = features_df.iloc[[i]]
            result = self.predict(url, row_df, threshold=threshold, explain=explain)
            result["url"] = url
            results.append(result)
        return results
