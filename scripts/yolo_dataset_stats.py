"""YOLO 데이터셋 이미지/라벨 개수 (로컬 실행용)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from yolo_dataset_utils import print_dataset_report  # noqa: E402


def main() -> None:
    rows = print_dataset_report(
        config.YOLO_DATA_YAML,
        fallback=config.HUMAN_FACE_DATASET,
    )
    if not rows:
        raise SystemExit(1)
    if sum(r["images"] for r in rows) == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
