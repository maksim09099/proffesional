from __future__ import annotations

import torch
import torch.nn as nn


class SimpleFaceNet(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        emb = self.embedding(x)
        logits = self.classifier(emb)
        return logits, emb


def save_checkpoint(path, model, label_to_idx: dict[str, int]):
    torch.save({"state_dict": model.state_dict(), "label_to_idx": label_to_idx}, path)
