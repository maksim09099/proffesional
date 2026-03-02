from __future__ import annotations

import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from src.models.dataset import FaceDataset
from src.models.inference import _load_model
from src.utils.config import load_config


def evaluate_model(config_path: str | Path = "configs/config.yaml") -> dict:
    cfg = load_config(config_path)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    model_path = Path(cfg["paths"]["models_dir"]) / "face_classifier.pt"

    checkpoint = torch.load(model_path, map_location="cpu")
    label_to_idx = checkpoint["label_to_idx"]
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    val_dir = processed_dir / "val"
    ds = FaceDataset(val_dir, label_to_idx)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    model, _ = _load_model(model_path)
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            logits, _ = model(x)
            preds = logits.argmax(dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())

    if not y_true:
        result = {"accuracy": 0.0, "report": "No validation samples"}
    else:
        result = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "report": classification_report(
                y_true,
                y_pred,
                target_names=[idx_to_label[i] for i in range(len(idx_to_label))],
                zero_division=0,
            ),
        }

    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    print(evaluate_model())
