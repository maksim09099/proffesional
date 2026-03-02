from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_model_info():
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert "model_path" in body
    assert "exists" in body


def test_predict(sample_image):
    with open(sample_image, "rb") as f:
        resp = client.post("/predict", files={"file": ("sample.jpg", f, "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    assert "predictions" in body or "error" in body
