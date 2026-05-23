"""학습용 ipynb를 notebooks/ 아래에 생성 (1회 실행용)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACE_ATTR = Path(r"C:\Users\moonjintae\projects\face_attr_mtl\notebooks\yolo_train_and_tune.ipynb")


def find_root() -> Path:
    return ROOT


def copy_yolo() -> Path:
    dst = ROOT / "notebooks" / "yolo" / "yolo_train_and_tune.ipynb"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if FACE_ATTR.is_file():
        shutil.copy2(FACE_ATTR, dst)
    return dst


def write_mtl() -> Path:
    path = ROOT / "notebooks" / "mtl" / "mtl_train_and_tune.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    # generated below via main
    return path


if __name__ == "__main__":
    yolo = copy_yolo()
    print("yolo:", yolo, yolo.stat().st_size if yolo.is_file() else "missing")
