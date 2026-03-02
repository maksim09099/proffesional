from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.models.model import SimpleFaceNet, save_checkpoint


@pytest.fixture(scope="session", autouse=True)
def prepare_dummy_model():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "face_classifier.pt"

    if not model_path.exists():
        model = SimpleFaceNet(num_classes=2)
        label_to_idx = {"person_a": 0, "person_b": 1}
        save_checkpoint(model_path, model, label_to_idx)

    yield


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    # simple face-like drawing
    cv2.circle(img, (100, 100), 60, (0, 0, 0), 2)
    cv2.circle(img, (80, 85), 8, (0, 0, 0), -1)
    cv2.circle(img, (120, 85), 8, (0, 0, 0), -1)
    cv2.ellipse(img, (100, 120), (25, 10), 0, 0, 180, (0, 0, 0), 2)

    image_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(image_path), img)
    return image_path
