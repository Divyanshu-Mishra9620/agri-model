from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import serve

class FakePredictor:
    """Stands in for DiseasePredictor so these tests don't load the real ~110MB model."""

    def __init__(self, *_args, **_kwargs):
        self.class_to_idx = {"Tomato_Early_blight": 0, "healthy": 1}
        self.device = "cpu"
        self.backend = "fake"

    def predict(self, image, top_k=3):
        return {
            "disease": "Tomato_Early_blight",
            "confidence": 91.2,
            "top_k": [{"disease": "Tomato_Early_blight", "confidence": 91.2}],
            "inference_time_ms": 1.0,
        }

def _fake_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(10, 120, 40)).save(buf, format="JPEG")
    return buf.getvalue()

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("inference.predictor.DiseasePredictor", FakePredictor)
    monkeypatch.setattr(serve, "ML_SERVICE_API_KEY", "test-key")
    with TestClient(serve.app) as c:
        yield c

def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["num_classes"] == 2

def test_predict_without_api_key_is_rejected(client):
    resp = client.post("/predict", files={"file": ("leaf.jpg", _fake_jpeg_bytes(), "image/jpeg")})
    assert resp.status_code == 401

def test_predict_with_wrong_api_key_is_rejected(client):
    resp = client.post(
        "/predict",
        headers={"X-API-Key": "wrong"},
        files={"file": ("leaf.jpg", _fake_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 401

def test_predict_with_correct_key_returns_prediction(client):
    resp = client.post(
        "/predict",
        headers={"X-API-Key": "test-key"},
        files={"file": ("leaf.jpg", _fake_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["disease"] == "Tomato_Early_blight"
    assert data["confidence"] == 91.2
    assert data["top_k"][0]["disease"] == "Tomato_Early_blight"

def test_predict_rejects_non_image_content_type(client):
    resp = client.post(
        "/predict",
        headers={"X-API-Key": "test-key"},
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400

def test_predict_rejects_undecodable_image_bytes(client):
    resp = client.post(
        "/predict",
        headers={"X-API-Key": "test-key"},
        files={"file": ("fake.jpg", b"not actually an image", "image/jpeg")},
    )
    assert resp.status_code == 400

def test_health_reports_degraded_when_model_fails_to_load(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated model load failure")

    monkeypatch.setattr("inference.predictor.DiseasePredictor", _raise)

    with TestClient(serve.app) as c:
        resp = c.get("/health")
        assert resp.json()["status"] == "degraded"
        assert resp.json()["model_loaded"] is False
