"""YOLO data.yaml 경로 해석 및 이미지/라벨 개수 (노트북·CLI 공용)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Roboflow / Ultralytics에서 자주 쓰는 split·폴더 조합
SPLIT_IMAGE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "train": ("train/images", "train", "images/train"),
    "val": ("valid/images", "val/images", "validation/images", "valid", "val"),
}


def count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(
        1
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def count_labels(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for _ in folder.rglob("*.txt"))


def labels_dir_for_images(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        return Path(*parts[:idx], "labels", *parts[idx + 1 :])
    return images_dir.parent / "labels"


def resolve_dataset_root(
    cfg: dict[str, Any],
    yaml_path: Path,
    *,
    fallback: Path | None = None,
) -> Path:
    raw = Path(str(cfg.get("path", ".")).strip())
    if raw.is_absolute():
        root = raw
    else:
        root = (yaml_path.parent / raw).resolve()

    if root.is_dir():
        return root

    if fallback is not None:
        fb = fallback.resolve()
        if fb.is_dir():
            return fb

    return root


def split_image_rel(cfg: dict[str, Any], split: str) -> str | None:
    if split == "train":
        return cfg.get("train")
    if split in ("val", "valid"):
        return cfg.get("val") or cfg.get("valid")
    return cfg.get(split)


def find_images_dir(root: Path, split: str, cfg_rel: str | None) -> Path:
    if cfg_rel:
        candidate = root / cfg_rel
        if candidate.is_dir():
            return candidate

    for rel in SPLIT_IMAGE_CANDIDATES.get(split, ()):
        candidate = root / rel
        if candidate.is_dir():
            return candidate

    return root / (cfg_rel or SPLIT_IMAGE_CANDIDATES[split][0])


def load_cfg(yaml_path: Path) -> dict[str, Any]:
    return yaml.safe_load(yaml_path.read_text(encoding="utf-8"))


def summarize_splits(
    yaml_path: Path,
    *,
    fallback: Path | None = None,
) -> list[dict[str, Any]]:
    cfg = load_cfg(yaml_path)
    root = resolve_dataset_root(cfg, yaml_path, fallback=fallback)
    rows: list[dict[str, Any]] = []

    for split in ("train", "val"):
        rel = split_image_rel(cfg, split)
        img_dir = find_images_dir(root, split, rel)
        lbl_dir = labels_dir_for_images(img_dir)
        rows.append(
            {
                "split": split,
                "root": root,
                "images_dir": img_dir,
                "labels_dir": lbl_dir,
                "images": count_images(img_dir),
                "labels": count_labels(lbl_dir),
                "images_exists": img_dir.is_dir(),
                "labels_exists": lbl_dir.is_dir(),
            }
        )
    return rows


def print_dataset_report(
    yaml_path: Path,
    *,
    fallback: Path | None = None,
    sample: int = 3,
) -> list[dict[str, Any]]:
    yaml_path = yaml_path.resolve()
    print(f"data.yaml: {yaml_path}")
    if not yaml_path.is_file():
        print("  → 없음. 실행: python scripts/link_human_face_dataset.py")
        return []

    cfg = load_cfg(yaml_path)
    root = resolve_dataset_root(cfg, yaml_path, fallback=fallback)
    print(f"cfg['path']: {cfg.get('path')!r}")
    print(f"resolved root: {root} (exists={root.is_dir()})")
    if fallback and fallback.resolve() != root:
        print(f"fallback (HUMAN_FACE_DATASET): {fallback.resolve()} (exists={fallback.is_dir()})")
    print()

    rows = summarize_splits(yaml_path, fallback=fallback)
    for row in rows:
        print(
            f"{row['split']}: {row['images']} images, {row['labels']} labels"
        )
        print(f"  images: {row['images_dir']} (dir={row['images_exists']})")
        print(f"  labels: {row['labels_dir']} (dir={row['labels_exists']})")
        if row["images"] == 0 and row["images_exists"]:
            print("  ⚠ 폴더는 있으나 이미지 0 — 확장자·OneDrive ‘항상 이 디바이스에 유지’ 확인")
        elif row["images"] == 0:
            print("  ⚠ images 폴더 없음 — data.yaml의 path/train·val 또는 데이터셋 구조 확인")
        else:
            samples = [
                p.name
                for p in row["images_dir"].rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ][:sample]
            if samples:
                print(f"  sample: {samples}")
        print()

    total_img = sum(r["images"] for r in rows)
    if total_img == 0:
        print(
            "진단: 프로젝트 data/face_detect/images 가 아니라 "
            "data.yaml의 path 아래 train·valid/images 를 봅니다.\n"
            "  1) python scripts/link_human_face_dataset.py\n"
            "  2) 노트북 커널 재시작 후 셀 1부터 다시 실행\n"
            "  3) 탐색기에서 train\\images 에 .jpg 가 있는지 확인"
        )
    return rows
