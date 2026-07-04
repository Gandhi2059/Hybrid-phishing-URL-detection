"""
PhishDetect — Core Library
==========================
Reusable modules for the phishing URL detection research project.

Sub-modules
-----------
preprocessing       : Data loading and cleaning utilities
feature_engineering : Lexical, security, statistical & TF-IDF feature extraction
rule_engine         : Heuristic rule-based URL scoring engine
hybrid_model        : Hybrid ML + rule-based prediction system with SHAP XAI
logistic_scratch    : Custom logistic regression implemented from scratch
evaluation          : Metrics, plotting (ROC, CM, bar charts), CSV export
"""

from .preprocessing import load_data, clean_data
from .feature_engineering import FeatureExtractor
from .rule_engine import RuleEngine
from .hybrid_model import HybridPhishingModel
from .logistic_scratch import LogisticRegressionScratch
from .evaluation import (
    evaluate_model,
    plot_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_mean_roc_curve,
    plot_rule_weight_sweep,
    save_metrics_to_csv,
)

__all__ = [
    "load_data",
    "clean_data",
    "FeatureExtractor",
    "RuleEngine",
    "HybridPhishingModel",
    "LogisticRegressionScratch",
    "evaluate_model",
    "plot_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_mean_roc_curve",
    "plot_rule_weight_sweep",
    "save_metrics_to_csv",
]
