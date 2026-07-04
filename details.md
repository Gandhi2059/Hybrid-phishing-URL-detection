# Missing Information for IEEE Publication Readiness

To ensure the project meets the standards of an IEEE conference or journal submission, the following information and tasks need to be addressed:

## 1. Literature Review & Citations
- **Missing:** A "Related Work" section with at least 15-20 references to existing research (2020-2024).
- **Need:** Specific comparisons with state-of-the-art models like PhishNet, URLNet, or recent Transformer-based (BERT/GPT) URL detectors.
- **Status:** Current repository documentation describes the system but does not include a formal survey of recent phishing URL detection architectures.

## 2. Comprehensive Experimental Validation
- **Missing:** Cross-validation (K-Fold) results. Currently, only a single 80/20 split is used.
- **Missing:** ROC (Receiver Operating Characteristic) and AUC (Area Under Curve) metrics. IEEE papers strictly require AUC-ROC curves.
- **Missing:** Latency benchmarks (e.g., average inference time in milliseconds per URL).
- **Status:** `run_pipeline.py` and `evaluation.py` evaluate models on a single split with accuracy, precision, recall, F1, and confusion matrices only.

## 3. Hardware & Environment Specifications
- **Missing:** Details of the hardware used for training (CPU model, RAM, GPU if any).
- **Missing:** Software environment details beyond the `requirements.txt` (operating system version, Python version, kernel information).
- **Status:** No environment capture or log of system hardware/software configuration is present in the current pipeline.

## 4. Theoretical Justification
- **Missing:** Justification for the `rule_weight` (0.15). Why was this specific value chosen?
- **Missing:** Formal proof or mathematical model of the hybrid fusion strategy beyond the current bounded addition.
- **Status:** `hybrid_model.py` contains the equation `hybrid_prob = min(ml_prob + (rule_score * self.rule_weight), 1.0)` but lacks sensitivity analysis or boundary proofs.

## 5. Rule-Weight Sensitivity & Error Distribution
- **Missing:** Analysis showing how varying `rule_weight` from 0.0 to 1.0 shifts False Positive and False Negative distributions.
- **Status:** Current codebase uses hardcoded `rule_weight` values in `app.py` and `api.py`, with no sweep study or plotted FP/FN trends.

## 6. Comparative Implementation Validation
- **Missing:** Direct comparison of the custom `LogisticRegressionScratch` vs. `sklearn.linear_model.LogisticRegression` to prove correctness.
- **Status:** `run_pipeline.py` trains the scratch implementation but does not compare it against the scikit-learn equivalent.

## 7. Real-World Testing
- **Missing:** Performance metrics on "zero-day" URLs (URLs generated after dataset collection date).
- **Status:** Data generation is synthetic/sample-based; no out-of-sample temporal validation or post-collection URL dataset is present.

## 8. Visual Assets for Paper
- **Missing:** High-resolution vector graphics (SVG/PDF) of the system architecture and ROC figures.
- **Status:** Existing graphs are PNG output only.

## 9. Limitations & Scope
- **Missing:** A quantitative analysis of adversarial attacks—how easily can an attacker bypass the hybrid engine by modifying the URL text?
- **Status:** The repository documents challenge scope qualitatively, but does not quantify adversarial robustness.

## Recommended Implementation Additions
- Extend the evaluation pipeline to perform stratified 10-fold cross-validation and compute ROC/AUC per fold.
- Export ROC curves and final evaluation figures in vector format (`SVG` or `PDF`).
- Add environment capture in the main pipeline for CPU model, RAM capacity, and OS/kernel version.
- Add a rule-weight sweep analysis from `0.0` to `1.0`, recording FP, FN, accuracy, and AUC to support the chosen `rule_weight` value.
- Add a `Related Work` section to the IEEE manuscript documentation with explicit references to modern phishing URL detectors.
