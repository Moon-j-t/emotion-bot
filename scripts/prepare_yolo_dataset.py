"""
원시 어노테이션 → Ultralytics YOLO 형식 데이터셋 변환.

지원:
  1) CSV (픽셀 bbox): image_path,xmin,ymin,xmax,ymax
  2) 기존 YOLO 폴더 (images/ + labels/) → train/val 분할
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _xyxy_to_yolo_line(
    xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int
) -> str:
    xmin = max(0.0, min(img_w, xmin))
    xmax = max(0.0, min(img_w, xmax))
    ymin = max(0.0, min(img_h, ymin))
    ymax = max(0.0, min(img_h, ymax))
    bw = max(0.0, xmax - xmin)
    bh = max(0.0, ymax - ymin)
    if bw <= 1 or bh <= 1:
        return ""
    xc = (xmin + xmax) / 2.0 / img_w
    yc = (ymin + ymax) / 2.0 / img_h
    w = bw / img_w
    h = bh / img_h
    return f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def _read_image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as img:
        return img.size  # (w, h)


def _clear_split_dirs(dataset_root: Path) -> None:
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = dataset_root / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)


def _write_data_yaml(dataset_root: Path) -> None:
    yaml_path = dataset_root / "data.yaml"
    content = {
        "path": str(dataset_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: config.YOLO_CLASS_NAME},
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False, allow_unicode=True)
    print(f"Wrote {yaml_path}")


def _copy_pair(
    image_src: Path,
    label_lines: list[str],
    dataset_root: Path,
    split: str,
) -> None:
    rel_name = image_src.name
    dst_img = dataset_root / "images" / split / rel_name
    dst_lbl = dataset_root / "labels" / split / f"{image_src.stem}.txt"
    shutil.copy2(image_src, dst_img)
    dst_lbl.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")


def prepare_from_csv(
    csv_path: Path,
    images_dir: Path,
    dataset_root: Path,
    val_ratio: float,
    seed: int,
) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {"image_path", "xmin", "ymin", "xmax", "ymax"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    groups: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for _, row in df.iterrows():
        groups[str(row["image_path"])].append(
            (float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"]))
        )

    items: list[tuple[Path, list[str]]] = []
    for rel, boxes in groups.items():
        img_path = images_dir / rel
        if not img_path.is_file():
            img_path = images_dir / Path(rel).name
        if not img_path.is_file():
            print(f"skip missing image: {rel}")
            continue

        w, h = _read_image_size(img_path)
        lines: list[str] = []
        for xmin, ymin, xmax, ymax in boxes:
            line = _xyxy_to_yolo_line(xmin, ymin, xmax, ymax, w, h)
            if line:
                lines.append(line)
        if lines:
            items.append((img_path, lines))

    _split_and_write(items, dataset_root, val_ratio, seed)


def prepare_from_yolo_source(source_root: Path, dataset_root: Path, val_ratio: float, seed: int) -> None:
    images_candidates = [
        source_root / "images",
        source_root,
    ]
    labels_candidates = [
        source_root / "labels",
        source_root,
    ]

    image_root = next((p for p in images_candidates if p.is_dir()), None)
    if image_root is None:
        raise FileNotFoundError(f"No images directory under {source_root}")

    items: list[tuple[Path, list[str]]] = []
    for img_path in sorted(image_root.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        label_path = None
        for lbl_root in labels_candidates:
            candidate = lbl_root / f"{img_path.stem}.txt"
            if candidate.is_file():
                label_path = candidate
                break
            candidate = lbl_root / img_path.relative_to(image_root).with_suffix(".txt")
            if candidate.is_file():
                label_path = candidate
                break

        if label_path is None:
            continue

        lines = [
            ln.strip()
            for ln in label_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if lines:
            items.append((img_path, lines))

    _split_and_write(items, dataset_root, val_ratio, seed)


def _split_and_write(
    items: list[tuple[Path, list[str]]],
    dataset_root: Path,
    val_ratio: float,
    seed: int,
) -> None:
    if not items:
        raise RuntimeError("No valid image/label pairs found.")

    random.seed(seed)
    random.shuffle(items)
    val_count = max(1, int(len(items) * val_ratio)) if len(items) > 1 else 0
    val_set = set(range(val_count))

    _clear_split_dirs(dataset_root)
    for i, (img_path, lines) in enumerate(items):
        split = "val" if i in val_set else "train"
        _copy_pair(img_path, lines, dataset_root, split)

    print(f"train={len(items) - val_count}, val={val_count}, total={len(items)}")
    _write_data_yaml(dataset_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO face detection dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DATA_DIR,
        help="Output dataset root (contains data.yaml)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=Path, default=None, help="CSV with pixel bboxes")
    parser.add_argument("--images-dir", type=Path, default=None, help="Image root for CSV mode")
    parser.add_argument(
        "--source-yolo",
        type=Path,
        default=None,
        help="Existing YOLO-style folder (images + labels)",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.csv:
        if not args.images_dir:
            raise SystemExit("--images-dir is required with --csv")
        prepare_from_csv(args.csv, args.images_dir, args.output, args.val_ratio, args.seed)
    elif args.source_yolo:
        prepare_from_yolo_source(args.source_yolo, args.output, args.val_ratio, args.seed)
    else:
        raise SystemExit("Provide --csv + --images-dir OR --source-yolo")

    print(f"Dataset ready at: {args.output.resolve()}")
    print("Next: python train/train_yolo_face.py")


if __name__ == "__main__":
    main()
