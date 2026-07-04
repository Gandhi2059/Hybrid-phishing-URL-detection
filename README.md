# PhishDetect 🛡️

> **Hybrid ML + Rule-Based Phishing URL Detection** — IEEE Research Project

PhishDetect is a research-grade phishing URL detection system that fuses
machine learning classifiers with a heuristic rule engine and SHAP-based
explainable AI (XAI). It ships with a FastAPI backend, a React/Vite frontend,
and a Streamlit diagnostic UI.

---

## Project Structure

```
phishing_detection_project/
│
├── src/                          # Core Python library (importable package)
│   ├── __init__.py
│   ├── preprocessing.py          # Data loading & cleaning
│   ├── feature_engineering.py    # Lexical + security + TF-IDF features
│   ├── rule_engine.py            # Heuristic rule-based URL scorer
│   ├── hybrid_model.py           # ML + rules fusion with SHAP XAI
│   ├── logistic_scratch.py       # Logistic regression from scratch (NumPy)
│   └── evaluation.py             # Metrics, ROC/CM plots, CSV export
│
├── scripts/                      # Runnable pipeline scripts
│   ├── prep_data.py              # Merge & deduplicate raw datasets → data/urls.csv
│   ├── train_rf.py               # Train Random Forest
│   ├── train_svm.py              # Train SVM (LinearSVC + calibration)
│   ├── train_xgb.py              # Train XGBoost
│   ├── train_all.py              # Train all models + 5-fold CV evaluation
│   ├── run_pipeline.py           # Convenience alias for train_all.py
│   └── get_acc.py                # Quick accuracy sanity-check
│
├── backend/                      # FastAPI REST API
│   └── api.py
│
├── frontend/                     # React + Vite web interface
│   ├── src/
│   ├── package.json
│   └── vite.config.js
│
├── data/                         # Raw and processed datasets (not committed)
│   ├── phishing_site_urls.csv    # Source dataset 1
│   ├── urldata.csv               # Source dataset 2
│   ├── urls.csv                  # Merged canonical dataset (generated)
│   └── features.csv              # Engineered features (generated)
│
├── models/                       # Trained model artifacts (not committed)
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   ├── xgb_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── graphs/                       # Generated plots (PNG / SVG / PDF)
│
├── ieee_paper/                   # IEEE manuscript sections
│   └── methodology.md
│
├── app.py                        # Streamlit UI (run from project root)
├── start_software.sh             # One-command full-stack launcher
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1 — Set up the environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Prepare the data

```bash
python scripts/prep_data.py
```

Merges `data/phishing_site_urls.csv` and `data/urldata.csv` into
`data/urls.csv` (~550 k deduplicated rows).

### 3 — Train all models

```bash
python scripts/train_all.py
```

Flags:

| Flag | Effect |
|---|---|
| `--rebuild-features` | Re-extract features from `data/urls.csv` |
| `--cv-only` | Skip training; re-run CV on existing models |
| `--no-cv` | Train only; skip cross-validation |
| `--cv-folds N` | Use N-fold CV (default: 5) |

### 4 — Launch the full-stack application

```bash
chmod +x start_software.sh
./start_software.sh
```

| Service | URL |
|---|---|
| **FastAPI backend** | http://127.0.0.1:8000/docs |
| **React frontend** | http://127.0.0.1:5173 |

Or launch the **Streamlit UI** only:

```bash
streamlit run app.py
```

### 5 — Quick accuracy check

```bash
python scripts/get_acc.py
python scripts/get_acc.py --n-samples 5000
```

---

## System Architecture

```
Raw CSVs  ──►  prep_data.py  ──►  urls.csv
                                      │
                              feature_engineering
                                      │
                              features.csv (cache)
                                      │
         ┌────────────────────────────┼──────────────────┐
         │                            │                  │
     train_rf.py               train_xgb.py         train_svm.py
         │                            │                  │
     rf_model.pkl             xgb_model.pkl        svm_model.pkl
                                      │
                          HybridPhishingModel
                         ┌─────────────────────┐
                         │ ML probability       │
                         │ + RuleEngine score   │  ──► final_score
                         │ + SHAP XAI reasons   │
                         └─────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                  │
              FastAPI API       Streamlit UI       React Frontend
```

---

## Hybrid Fusion Formula

```
hybrid_prob = min(ml_prob + rule_score × rule_weight, 1.0)
```

- `ml_prob` — P(phishing) from XGBoost (default serving model)  
- `rule_score` — integer penalty from the RuleEngine (0–20)  
- `rule_weight` — 0.15 (empirically chosen; see `details.md` §4)

---

## Models & Performance (5-fold Stratified CV)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| XGBoost | ~0.967 | ~0.968 | ~0.967 | ~0.967 | ~0.992 |
| Random Forest | ~0.963 | ~0.964 | ~0.963 | ~0.963 | ~0.990 |
| SVM (LinearSVC) | ~0.958 | ~0.959 | ~0.958 | ~0.958 | — |
| Logistic Reg. (scratch) | ~0.931 | ~0.931 | ~0.931 | ~0.931 | — |

> Exact values depend on dataset split and environment. Check `graphs/metrics_summary.csv` after training.

---

## API Reference

### `POST /api/detect`

```json
{ "url": "http://secure-login.bank-update.com" }
```

**Response:**

```json
{
  "url": "...",
  "prediction": "Phishing",
  "final_score": 0.94,
  "ml_probability": 0.87,
  "rule_score": 5.0,
  "rule_details": ["No HTTPS (+2)", "Keyword 'login' (+2)", "Keyword 'bank' (+2)"],
  "xai_reasons": ["Url Length (SHAP: +0.21)", "Has Https (SHAP: +0.18)"],
  "threat_intel": { "dns_resolves": false, "domain_intel": "DNS resolution failed" }
}
```

### `POST /api/scan_batch`

```json
{ "urls": ["http://phish.example.com", "https://google.com"] }
```

Returns `{ "job_id": "...", "status": "processing", "queued_count": 2 }`.
Poll with `GET /api/job/{job_id}`.

---

## Development

```bash
# Run backend with auto-reload
uvicorn backend.api:app --reload

# Run Streamlit UI
streamlit run app.py

# Run frontend dev server
cd frontend && npm run dev
```

---

## Research Context

This project is developed for an IEEE conference/journal paper on
hybrid phishing URL detection. See:



---

## License

Academic / research use only. Not licensed for commercial deployment.
