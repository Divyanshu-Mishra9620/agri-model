"""
FastAPI serving layer for the trained disease classifier.

This repo (see README.md > "Integration with KrishiNova") trains and exports
the model but deliberately does not serve it as part of the training
pipeline. This module is that missing serving layer: it loads
`inference.predictor.DiseasePredictor` ONCE at startup and exposes it over
HTTP so the RAG service (`Retrieval aug gen/rag`) can call it per-request
instead of re-loading a ~110MB model on every image.

This process is a private, service-to-service API — it is called by the RAG
backend, never directly by a browser — so it authenticates callers with a
shared API key (ML_SERVICE_API_KEY) rather than the farmer-facing user JWT,
and does not need CORS.

Run from this directory (ml/), with the same venv used for training/export:
    uvicorn serve:app --host 0.0.0.0 --port 8500

Environment variables (see .env.example):
    ML_MODEL_PATH        path to the exported model (default: outputs/export/model.torchscript.pt)
    ML_SERVICE_API_KEY    shared secret required on the X-API-Key header
    ML_SERVICE_PORT       default 8500 (only used by `python serve.py` directly)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ml.serve")

ML_MODEL_PATH = Path(os.getenv("ML_MODEL_PATH", "outputs/export/model.torchscript.pt"))
ML_SERVICE_API_KEY = os.getenv("ML_SERVICE_API_KEY", "")
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB, matches the RAG/backend upload caps

if not ML_SERVICE_API_KEY:
    logger.error(
        "ML_SERVICE_API_KEY is not set. Every /predict request will be rejected "
        "(fail closed) until this is configured — it must match the value the "
        "RAG service is configured to send."
    )

# TorchScript inference (eval mode, no in-place state mutation) is safe to call
# concurrently in practice, but a lock removes any doubt for near-zero cost at
# this service's expected request volume — simplicity over marginal throughput.
_predict_lock = threading.Lock()


class TopKItem(BaseModel):
    disease: str
    confidence: float


class PredictResponse(BaseModel):
    disease: str
    confidence: float
    top_k: list[TopKItem]
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int | None = None
    device: str | None = None
    backend: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading DiseasePredictor from %s ...", ML_MODEL_PATH)
    try:
        from inference.predictor import DiseasePredictor

        app.state.predictor = DiseasePredictor(ML_MODEL_PATH)
        logger.info(
            "Model ready: %d classes, backend=%s, device=%s",
            len(app.state.predictor.class_to_idx),
            app.state.predictor.backend,
            app.state.predictor.device,
        )
    except Exception:
        logger.exception("Failed to load model at startup — /predict will return 503")
        app.state.predictor = None
    yield


app = FastAPI(
    title="KrishiNova Disease Classifier — Serving API",
    description="Internal service: predicts a crop disease class from an image.",
    version="1.0.0",
    lifespan=lifespan,
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not ML_SERVICE_API_KEY:
        raise HTTPException(status_code=503, detail="ML service authentication is not configured")
    if not x_api_key or x_api_key != ML_SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        num_classes=len(predictor.class_to_idx),
        device=str(predictor.device),
        backend=predictor.backend,
    )


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=10),
) -> PredictResponse:
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10MB size limit.")

    array_bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if array_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    array_rgb = cv2.cvtColor(array_bgr, cv2.COLOR_BGR2RGB)

    top_k = min(top_k, len(predictor.class_to_idx))

    def _run() -> dict:
        with _predict_lock:
            return predictor.predict(array_rgb, top_k=top_k)

    try:
        import asyncio

        result = await asyncio.to_thread(_run)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    return PredictResponse(**result)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("ML_SERVICE_PORT", 8500))
    uvicorn.run("serve:app", host="0.0.0.0", port=port, reload=False, log_level="info")
