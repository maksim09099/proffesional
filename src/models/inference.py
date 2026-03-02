from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import torch

from src.models.model import SimpleFaceNet


@dataclass
class Prediction:
    bbox: tuple[int, int, int, int]
    label: str
    confidence: float


def _load_model(model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu")
    label_to_idx = checkpoint["label_to_idx"]
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    model = SimpleFaceNet(num_classes=len(label_to_idx))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, idx_to_label


def predict_faces(image_path: str | Path, model_path: str | Path = "models/face_classifier.pt") -> dict[str, Any]:
    image_path = Path(image_path)
    model_path = Path(model_path)

    model, idx_to_label = _load_model(model_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))

    preds: list[Prediction] = []
    annotated = image.copy()
    for (x, y, w, h) in faces:
        crop = image[y:y + h, x:x + w]
        crop = cv2.resize(crop, (160, 160))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            logits, _ = model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, cls = probs.max(dim=1)

        label = idx_to_label[int(cls.item())]
        confidence = float(conf.item())
        preds.append(Prediction((int(x), int(y), int(w), int(h)), label, confidence))
        cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{label} {confidence:.2f}",
            (int(x), max(20, int(y) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    annotated_path = image_path.with_name(f"{image_path.stem}_annotated{image_path.suffix}")
    cv2.imwrite(str(annotated_path), annotated)

    return {
        "image": str(image_path),
        "num_faces": len(preds),
        "predictions": [
            {"bbox": p.bbox, "class": p.label, "confidence": p.confidence} for p in preds
        ],
        "annotated_image": str(annotated_path),
    }
