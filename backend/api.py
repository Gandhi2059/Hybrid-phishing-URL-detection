"""
backend/api.py
--------------
FastAPI REST backend for the PhishDetect phishing URL detection system.

Endpoints
---------
POST /api/detect          — Single URL analysis (ML + rules + XAI + threat intel)
POST /api/scan_batch      — Async bulk URL scanning (returns job_id)
GET  /api/job/{job_id}    — Poll an async batch job for results
GET  /api/health          — Liveness probe

Run
---
    # from project root
    uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import os
import pickle
import socket
import sys
import uuid
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from cachetools import TTLCache
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from urllib.parse import urlparse

from src.feature_engineering import FeatureExtractor
from src.hybrid_model import HybridPhishingModel

# ── App setup ─────────────────────────────────────────────────────────────── #
app = FastAPI(
    title="PhishDetect API",
    description="Hybrid ML + heuristic phishing URL detection with SHAP XAI.",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Threat intel cache (DNS / WHOIS simulation) ───────────────────────────── #
_intel_cache: TTLCache = TTLCache(maxsize=1000, ttl=600)


class ThreatIntel:
    """Lightweight DNS-based domain enrichment with TTL-aware caching."""

    @staticmethod
    async def enrich_domain(url: str) -> dict:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        if not domain:
            return {"dns_resolves": False, "domain_intel": "Invalid domain"}

        if domain in _intel_cache:
            return _intel_cache[domain]

        await asyncio.sleep(0.05)          # simulate async IO latency
        try:
            socket.gethostbyname(domain)
            result = {"dns_resolves": True,  "domain_intel": "Domain active"}
        except socket.error:
            result = {"dns_resolves": False, "domain_intel": "DNS resolution failed"}

        _intel_cache[domain] = result
        return result


# ── Model loading ─────────────────────────────────────────────────────────── #
ROOT        = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR  = os.path.join(ROOT, "models")


class _DummyModel:
    """Fallback when a trained .pkl is not yet available."""
    def predict_proba(self, X):
        return np.array([[0.1, 0.9]])


def _load_model(name: str):
    path = os.path.join(MODELS_DIR, name)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"⚠️  {path} not found — using dummy model. Run scripts/train_all.py.")
        return _DummyModel()


ml_model = _load_model("xgb_model.pkl")

extractor      = FeatureExtractor(use_tfidf=True, max_tfidf_features=100)
extractor.load_vectorizer(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

hybrid_system  = HybridPhishingModel(ml_model=ml_model, rule_weight=0.15)

# ── In-memory batch job store (use Celery/Redis in production) ────────────── #
_batch_jobs: dict[str, dict] = {}


# ── Pydantic schemas ─────────────────────────────────────────────────────── #
class URLRequest(BaseModel):
    url: str


class URLResponse(BaseModel):
    url: str
    prediction: str
    final_score: float
    ml_probability: float
    rule_score: float
    rule_details: list[str]
    xai_reasons: list[str] = []
    threat_intel: dict = {}


class BatchRequest(BaseModel):
    urls: List[str]


class BatchResponse(BaseModel):
    job_id: str
    status: str
    queued_count: int


# ── Helpers ──────────────────────────────────────────────────────────────── #
def _analyse(url: str) -> dict:
    """Feature extraction + hybrid prediction for a single URL."""
    df_single   = pd.DataFrame({"url": [url]})
    df_features = extractor.extract_features(df_single)
    X_input     = df_features.drop(["url"], axis=1)
    return hybrid_system.predict(url, X_input)


# ── Routes ───────────────────────────────────────────────────────────────── #
@app.post("/api/detect", response_model=URLResponse, tags=["Detection"])
async def detect_phishing(request: URLRequest):
    """Analyse a single URL for phishing indicators."""
    if not request.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty.")
    try:
        result     = _analyse(request.url)
        intel_data = await ThreatIntel.enrich_domain(request.url)
        return URLResponse(
            url=request.url,
            prediction=result["prediction"],
            final_score=result["final_score"],
            ml_probability=result["ml_probability"],
            rule_score=result["rule_score"],
            rule_details=result["rule_details"],
            xai_reasons=result.get("xai_reasons", []),
            threat_intel=intel_data,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


async def _process_batch(job_id: str, urls: List[str]) -> None:
    """Background coroutine that processes a batch scanning job."""
    results = []
    for url in urls:
        result              = _analyse(url)
        result["url"]       = url
        result["threat_intel"] = await ThreatIntel.enrich_domain(url)
        results.append(result)
    _batch_jobs[job_id] = {"status": "completed", "results": results}


@app.post("/api/scan_batch", response_model=BatchResponse, tags=["Detection"])
async def scan_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    """Submit a list of URLs for asynchronous bulk scanning."""
    job_id = str(uuid.uuid4())
    _batch_jobs[job_id] = {"status": "processing", "results": []}
    background_tasks.add_task(_process_batch, job_id, request.urls)
    return BatchResponse(job_id=job_id, status="processing", queued_count=len(request.urls))


@app.get("/api/job/{job_id}", tags=["Detection"])
async def get_job_status(job_id: str):
    """Poll an async batch job for results."""
    if job_id not in _batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _batch_jobs[job_id]


@app.get("/api/health", tags=["System"])
async def health_check():
    """Liveness probe."""
    return {"status": "healthy", "version": app.version}
