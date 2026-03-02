from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import pandas as pd

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger


@dataclass
class AuditResult:
    total_images: int
    valid_images: int
    invalid_images: int
    missing_labels: int
    classes_found: int
    issues_log: Path
    report_csv: Path
    summary_json: Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def unpack_archives(raw_dir: Path, extracted_dir: Path) -> List[Path]:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    archive_names = ["Data1.zip", "Data2.zip", "Data3.zip", "Data4.zip"]
    unpacked: List[Path] = []

    for name in archive_names:
        archive_path = raw_dir / name
        if not archive_path.exists():
            continue
        target_dir = extracted_dir / archive_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(target_dir)
        unpacked.append(target_dir)

    return unpacked


def _collect_images(extracted_dir: Path) -> List[Path]:
    return [
        p
        for p in extracted_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def run_data_audit(config_path: str | Path = "configs/config.yaml") -> AuditResult:
    cfg = load_config(config_path)
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    extracted_dir = Path(cfg["paths"]["extracted_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    logger = setup_logger("data_audit", reports_dir / "data_audit.log")

    unpack_archives(raw_dir=raw_dir, extracted_dir=extracted_dir)
    images = _collect_images(extracted_dir)

    issues: List[Dict[str, str]] = []
    rows: List[Dict[str, str | int]] = []

    for image_path in images:
        label = image_path.parent.name.strip()
        img = cv2.imread(str(image_path))
        valid_image = img is not None

        if not valid_image:
            issues.append({"file": str(image_path), "issue": "Unreadable image"})

        if label == "" or label.lower() in {"unknown", "unlabeled", "misc"}:
            issues.append({"file": str(image_path), "issue": "Missing or weak label"})

        rows.append(
            {
                "file": str(image_path),
                "label": label,
                "is_valid_image": int(valid_image),
                "has_label": int(label != ""),
            }
        )

    df = pd.DataFrame(rows)
    report_csv = reports_dir / "data_audit_report.csv"
    issues_json = reports_dir / "data_issues.json"
    summary_json = reports_dir / "data_audit_summary.json"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(report_csv, index=False)
    with issues_json.open("w", encoding="utf-8") as f:
        json.dump(issues, f, ensure_ascii=False, indent=2)

    total = int(len(df))
    valid = int(df["is_valid_image"].sum()) if total else 0
    missing_labels = int((df["has_label"] == 0).sum()) if total else 0
    classes_found = int(df["label"].nunique()) if total else 0

    summary = {
        "total_images": total,
        "valid_images": valid,
        "invalid_images": total - valid,
        "missing_labels": missing_labels,
        "classes_found": classes_found,
        "issues_count": len(issues),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Data audit completed: %s", summary)

    return AuditResult(
        total_images=summary["total_images"],
        valid_images=summary["valid_images"],
        invalid_images=summary["invalid_images"],
        missing_labels=summary["missing_labels"],
        classes_found=summary["classes_found"],
        issues_log=issues_json,
        report_csv=report_csv,
        summary_json=summary_json,
    )


if __name__ == "__main__":
    result = run_data_audit()
    print(result)
