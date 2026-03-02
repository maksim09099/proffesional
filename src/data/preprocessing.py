from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger


def _detect_largest_face(image, face_cascade, min_size: int) -> Tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
    return int(x), int(y), int(w), int(h)


def build_processed_dataset(config_path: str | Path = "configs/config.yaml") -> Dict[str, int]:
    cfg = load_config(config_path)
    extracted_dir = Path(cfg["paths"]["extracted_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    image_size = int(cfg["preprocessing"]["image_size"])
    min_face_size = int(cfg["preprocessing"]["min_face_size"])
    train_split = float(cfg["preprocessing"]["train_split"])

    logger = setup_logger("preprocessing", reports_dir / "preprocessing.log")

    face_cascade = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )

    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    rows: List[Dict[str, str | int]] = []
    all_images = [
        p
        for p in extracted_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    valid_records: List[Tuple[Path, str, any]] = []
    for img_path in all_images:
        label = img_path.parent.name.strip()
        image = cv2.imread(str(img_path))
        if image is None:
            rows.append({"file": str(img_path), "label": label, "status": "unreadable"})
            continue

        face = _detect_largest_face(image, face_cascade, min_size=min_face_size)
        if face is None:
            rows.append({"file": str(img_path), "label": label, "status": "no_face"})
            continue

        x, y, w, h = face
        cropped = image[y:y + h, x:x + w]
        resized = cv2.resize(cropped, (image_size, image_size))
        valid_records.append((img_path, label, resized))

    if not valid_records:
        logger.warning("No valid face samples found.")
        return {"total": len(all_images), "processed": 0}

    labels = [r[1] for r in valid_records]
    idx = list(range(len(valid_records)))
    train_idx, val_idx = train_test_split(idx, train_size=train_split, stratify=labels, random_state=42)
    train_set, val_set = set(train_idx), set(val_idx)

    for i, (_, label, image) in enumerate(valid_records):
        split = "train" if i in train_set else "val"
        out_dir = processed_dir / split / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"sample_{i:06d}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), image)
        rows.append({"file": str(out_path), "label": label, "status": split})

    report_df = pd.DataFrame(rows)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(reports_dir / "preprocessing_report.csv", index=False)

    summary = {
        "total": len(all_images),
        "processed": len(valid_records),
        "train": len(train_set),
        "val": len(val_set),
    }
    with (reports_dir / "preprocessing_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Preprocessing completed: %s", summary)
    return summary


if __name__ == "__main__":
    print(build_processed_dataset())
