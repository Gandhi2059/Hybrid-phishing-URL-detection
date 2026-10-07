# PhishDetect

Hybrid phishing URL detection using machine-learning features, heuristic rules, and explainable model output. The repository includes a FastAPI service, a React/Vite web interface, a Streamlit interface, and scripts for preparing data, training models, and producing evaluation figures.

## What It Does

- Extracts URL length, punctuation and character counts, HTTPS/IP indicators, suspicious-keyword counts, character entropy, ratios, and character-level TF-IDF features.
- Combines an XGBoost phishing probability with a rule-engine score. The default rule weight is `0.15`; the fused score is capped at `1.0` and classified at a `0.5` threshold.
- Returns triggered rule explanations, optional SHAP feature contributions, and DNS-based domain status from the API.
- Supports single-URL analysis and asynchronous batch scans.

The rule engine assigns points for missing HTTPS, `@` in the URL, suspicious keywords, long URLs, IP addresses, and excess dots/subdomains. Treat results as research signals, not a substitute for security review.

## Repository Layout

```text
backend/       FastAPI application
data/          datasets and generated feature tables
frontend/      React 19 + Vite interface
graphs/        evaluation outputs and research figures
models/        trained model and TF-IDF artifacts
scripts/       data, training, evaluation, and figure-generation scripts
src/           feature engineering, rules, hybrid model, preprocessing, evaluation
app.py         Streamlit interface
kaggle_train.py  Kaggle-oriented training entry point
start_software.sh  launcher for the API and web interface
```

## Requirements

- Python 3.10 or newer
- Node.js and npm for the React frontend
- Git

Install the Python dependencies from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Model Files

Trained model files are not included in a fresh clone. Download the artifacts from the [model files folder](https://drive.google.com/drive/folders/1t4vNpcFBsSUKFyJM1AHC3y-hvOp33VTm?usp=sharing) and put them in `models/`:

```text
models/xgb_model.pkl
models/tfidf_vectorizer.pkl
```

The API uses XGBoost and the TF-IDF vectorizer for detection. Random Forest and SVM artifacts are used by the evaluation scripts. Model pickles can depend on the Python and scikit-learn versions used to create them; use compatible versions when loading them. Without the expected artifacts, the API falls back to a dummy model and its output is not a meaningful phishing prediction.

## Run The Application

Install frontend dependencies once:

```bash
cd frontend
npm install
cd ..
```

Start the FastAPI backend and React frontend from the project root:

```bash
./start_software.sh
```

Open the frontend at <http://127.0.0.1:5173> and the interactive API documentation at <http://127.0.0.1:8000/docs>. The launcher supports `--api-only` and `--ui-only` modes. It uses `fuser` to clear ports 8000 and 5173 before startup.

To run only the Streamlit interface instead:

```bash
streamlit run app.py
```

Or start the API and frontend separately:

```bash
# From the project root
uvicorn backend.api:app --host 127.0.0.1 --port 8000

# In another terminal
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

## API

### Detect one URL

`POST /api/detect`

```bash
curl -X POST http://127.0.0.1:8000/api/detect \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com/login"}'
```

The response includes `prediction`, `final_score`, `ml_probability`, `rule_score`, `rule_details`, `xai_reasons`, and `threat_intel`.

### Scan a batch

Submit a list of URLs to `POST /api/scan_batch`:

```bash
curl -X POST http://127.0.0.1:8000/api/scan_batch \
  -H 'Content-Type: application/json' \
  -d '{"urls":["https://example.com","http://suspicious.example/login"]}'
```

The response contains a `job_id`. Poll `GET /api/job/{job_id}` until its status is `completed` to retrieve results. `GET /api/health` returns the API health status.

The API performs DNS resolution for domain enrichment and allows all CORS origins in its current configuration. Restrict CORS and review network access before exposing it outside a trusted development environment.

## Data Preparation And Training

The raw training CSVs are excluded from Git. To rebuild the training dataset, place these files under `data/`:

- `phishing_site_urls.csv`, with `URL` and `Label` columns (`good`/`bad` labels)
- `urldata.csv`, with `url` and `label` columns (`benign`/`malicious` labels)

Then prepare the merged dataset and train/evaluate the models:

```bash
python scripts/prep_data.py
python scripts/train_all.py
```

`prep_data.py` writes a deduplicated, shuffled `data/urls.csv`. `train_all.py` builds `data/features.csv` if needed, saves model artifacts under `models/`, and by default runs stratified 5-fold cross-validation and writes metrics/plots under `graphs/`.

Useful options:

| Command | Behavior |
|---|---|
| `python scripts/train_all.py --rebuild-features` | Rebuild features from `data/urls.csv` |
| `python scripts/train_all.py --no-cv` | Train models without cross-validation |
| `python scripts/train_all.py --cv-only` | Run cross-validation without the full-data training step |
| `python scripts/train_all.py --cv-folds 10` | Use 10 cross-validation folds |
| `python scripts/evaluate_models.py --n-samples 10000` | Evaluate available models on a sample of `data/urls.csv` |
| `python scripts/evaluate_models.py --full` | Evaluate available models on the full dataset |

Training and full-dataset evaluation can require substantial memory and time. The scripts generate current metrics; results depend on the dataset, artifacts, and environment, so this README does not hard-code benchmark values.

## Research And Project Files

- [Zenodo project archive](https://doi.org/10.5281/zenodo.23141305)
- [Trained model files](https://drive.google.com/drive/folders/1t4vNpcFBsSUKFyJM1AHC3y-hvOp33VTm?usp=sharing)
- `details.md`, `graphs/`, and the figure-generation scripts contain supporting methodology and research outputs.

## License And Use

Academic/research use only. Not licensed for commercial deployment. This project is a research prototype and should not be treated as a production security service.