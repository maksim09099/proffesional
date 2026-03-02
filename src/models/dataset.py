from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, root_dir: Path, label_to_idx: dict[str, int]):
        self.samples: List[Tuple[Path, int]] = []
        self.root_dir = root_dir
        self.label_to_idx = label_to_idx

        for class_dir in sorted(root_dir.glob("*")):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            if label not in label_to_idx:
                continue
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((img_path, label_to_idx[label]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        return tensor, torch.tensor(label, dtype=torch.long)
