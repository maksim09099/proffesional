from __future__ import annotations

from src.models.inference import predict_faces


def test_predict_faces_returns_expected_keys(sample_image):
    result = predict_faces(sample_image)
    assert "num_faces" in result
    assert "predictions" in result
    assert "annotated_image" in result
