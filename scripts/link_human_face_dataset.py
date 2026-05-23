"""
Roboflow YOLO 데이터셋 → emotion-bot 학습용 data.yaml 연동.

기본 소스:
  C:\\Users\\moonjintae\\datasets\\Human Face dataset
  (train/images, train/labels, valid/images, valid/labels)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402

DEFAULT_SOURCE = Path(r"C:\Users\moonjintae\datasets\Human Face dataset")


def validate_roboflow_yolo(root: Path) -> None:
    required = [
        root / "train" / "images",
        root / "train" / "labels",
        root / "valid" / "images",
        root / "valid" / "labels",
    ]
    missing = [p for p in required if not p.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Roboflow YOLO 폴더 구조가 아닙니다. 다음이 필요합니다:\n"
            "  train/images, train/labels, valid/images, valid/labels\n"
            f"누락: {missing}"
        )


def write_project_data_yaml(source_root: Path, output_yaml: Path) -> Path:
    validate_roboflow_yolo(source_root)
    content = {
        "path": str(source_root.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "nc": 1,
        "names": {0: config.YOLO_CLASS_NAME},
    }
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with output_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False, allow_unicode=True)
    return output_yaml


def count_images(split_dir: Path) -> int:
    if not split_dir.is_dir():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in split_dir.iterdir() if p.suffix.lower() in exts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Link Human Face YOLO dataset")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Roboflow export root (contains train/, valid/)",
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=config.YOLO_DATA_YAML,
        help="Project data.yaml to write",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)

    out = write_project_data_yaml(source, args.output_yaml.resolve())
    train_n = count_images(source / "train" / "images")
    val_n = count_images(source / "valid" / "images")

    print(f"Linked dataset: {source}")
    print(f"Wrote: {out}")
    print(f"train images: {train_n}, valid images: {val_n}")
    print("Next: python train/train_yolo_face.py")


if __name__ == "__main__":
    main()
