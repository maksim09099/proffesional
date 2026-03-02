from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.dataset import FaceDataset
from src.models.model import SimpleFaceNet, save_checkpoint
from src.utils.config import load_config


def train_model(config_path: str | Path = "configs/config.yaml") -> dict:
    cfg = load_config(config_path)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])

    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    classes = sorted([p.name for p in train_dir.glob("*") if p.is_dir()])
    if not classes:
        raise RuntimeError("No training data found in data/processed/train")

    label_to_idx = {name: i for i, name in enumerate(classes)}
    train_ds = FaceDataset(train_dir, label_to_idx)
    val_ds = FaceDataset(val_dir, label_to_idx)

    batch_size = int(cfg["model"]["batch_size"])
    num_workers = int(cfg["model"]["num_workers"])
    lr = float(cfg["model"]["lr"])
    epochs = int(cfg["model"]["epochs"])
    embedding_dim = int(cfg["model"]["embedding_dim"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleFaceNet(num_classes=len(classes), embedding_dim=embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                preds = logits.argmax(dim=1)
                correct += int((preds == y).sum().item())
                total += int(y.size(0))

        val_acc = (correct / total) if total else 0.0
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(len(train_loader), 1),
            "val_acc": val_acc,
        }
        history.append(epoch_result)
        print(epoch_result)

    models_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(models_dir / "face_classifier.pt", model, label_to_idx)

    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return {
        "classes": classes,
        "epochs": epochs,
        "model_path": str(models_dir / "face_classifier.pt"),
        "history": history,
    }


if __name__ == "__main__":
    print(train_model())
