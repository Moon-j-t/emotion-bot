"""
Roboflow YOLO 데이터셋 → emotion-bot 학습용 data.yaml 연동.

기본 소스:
  C:\\Users\\moonjintae\\datasets\\Human Face dataset
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

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from yolo_dataset_utils import count_images, count_labels  # noqa: E402


def _yaml_content(source_root: Path) -> dict:
    return {
        "path": str(source_root.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "nc": 1,
        "names": {0: config.YOLO_CLASS_NAME},
    }


def link_dataset(source_root: Path, project_yaml: Path) -> None:
    source_root = source_root.resolve()
    train_img = source_root / "train" / "images"
    train_lbl = source_root / "train" / "labels"
    val_img = source_root / "valid" / "images"
    val_lbl = source_root / "valid" / "labels"

    if not train_lbl.is_dir() or not val_lbl.is_dir():
        raise FileNotFoundError(
            f"labels 폴더가 없습니다: {source_root}\n"
            "  필요: train/labels, valid/labels"
        )

    content = _yaml_content(source_root)
    project_yaml.parent.mkdir(parents=True, exist_ok=True)
    with project_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False, allow_unicode=True)

    source_yaml = source_root / "data.yaml"
    with source_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False, allow_unicode=True)

    print(f"Dataset root: {source_root}")
    print(f"Project yaml: {project_yaml}")
    print(f"Source yaml:  {source_yaml}")
    print(
        f"train: {count_images(train_img)} images, {count_labels(train_lbl)} labels"
    )
    print(f"val:   {count_images(val_img)} images, {count_labels(val_lbl)} labels")

    if count_images(train_img) == 0:
        print(
            "\n⚠ 이 PC에서 train/images 이미지 0개로 보입니다.\n"
            "  폴더에 파일이 있는데 0이면: python scripts/yolo_dataset_stats.py 로 재확인"
        )
    else:
        print("\nOK — python train/train_yolo_face.py 또는 YOLO 노트북으로 학습")


def main() -> None:
    parser = argparse.ArgumentParser(description="Link Human Face YOLO dataset")
    parser.add_argument("--source", type=Path, default=config.HUMAN_FACE_DATASET)
    parser.add_argument("--output-yaml", type=Path, default=config.YOLO_DATA_YAML)
    args = parser.parse_args()

    if not args.source.is_dir():
        raise FileNotFoundError(args.source)

    link_dataset(args.source, args.output_yaml.resolve())


if __name__ == "__main__":
    main()
