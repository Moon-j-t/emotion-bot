"""
Roboflow YOLO 표정 데이터셋 → MTL 학습용 (얼굴 크롭 + 표정/나이 CSV).

원본 예: Downloads/emotion detection.v1i.yolov8
  train/images, train/labels, valid/images, ...

나이 라벨: InsightFace (공개 사전학습, 얼굴 속성 추정)
  pip install insightface onnxruntime

출력: data/mtl/
  faces/train/*.jpg, faces/val/*.jpg
  train.csv, val.csv  (image_path, expression_id, age)
"""
from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Roboflow class name → config.EXPRESSION_LABELS index
# "content"는 7클래스에 없음 → neutral(6)로 매핑 (README에 명시)
ROBOFLOW_NAME_TO_EXPRESSION_ID: dict[str, int] = {
    "anger": 0,      # angry
    "angry": 0,
    "content": 6,    # neutral (만족/평온 계열)
    "contempt": 6,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "happiness": 3,
    "neutral": 6,
    "sad": 4,
    "sadness": 4,
    "surprise": 5,
}

SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "val": "val",
}


@dataclass
class Stats:
    images_seen: int = 0
    saved: int = 0
    skipped_no_label: int = 0
    skipped_no_box: int = 0
    skipped_small_crop: int = 0
    skipped_unknown_class: int = 0
    skipped_age_failed: int = 0
    class_counts: dict[int, int] = field(default_factory=dict)


def load_roboflow_names(source_root: Path) -> list[str]:
    yaml_path = source_root / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml 없음: {yaml_path}")
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = cfg.get("names")
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    if isinstance(names, list):
        return list(names)
    raise ValueError("data.yaml에 names가 없습니다.")


def parse_yolo_label_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    return (
        int(parts[0]),
        float(parts[1]),
        float(parts[2]),
        float(parts[3]),
        float(parts[4]),
    )


def yolo_box_to_pixels(
    xc: float, yc: float, bw: float, bh: float, img_w: int, img_h: int, padding: float
) -> tuple[int, int, int, int]:
    x1 = (xc - bw / 2) * img_w
    y1 = (yc - bh / 2) * img_h
    x2 = (xc + bw / 2) * img_w
    y2 = (yc + bh / 2) * img_h
    pad_w = (x2 - x1) * padding
    pad_h = (y2 - y1) * padding
    x1 = int(max(0, x1 - pad_w))
    y1 = int(max(0, y1 - pad_h))
    x2 = int(min(img_w, x2 + pad_w))
    y2 = int(min(img_h, y2 + pad_h))
    return x1, y1, x2, y2


def pick_best_box(lines: list[str]) -> tuple[int, float, float, float, float] | None:
    best: tuple[int, float, float, float, float] | None = None
    best_area = -1.0
    for line in lines:
        parsed = parse_yolo_label_line(line)
        if parsed is None:
            continue
        _, _, _, bw, bh = parsed
        area = bw * bh
        if area > best_area:
            best_area = area
            best = parsed
    return best


def class_id_to_expression_id(class_id: int, class_names: list[str]) -> int | None:
    if class_id < 0 or class_id >= len(class_names):
        return None
    name = class_names[class_id].strip().lower()
    if name in ROBOFLOW_NAME_TO_EXPRESSION_ID:
        return ROBOFLOW_NAME_TO_EXPRESSION_ID[name]
    # id 자체가 expression index와 같다고 가정 (0~6)
    if 0 <= class_id < config.NUM_EXPRESSIONS:
        return class_id
    return None


class AgeEstimator(Protocol):
    def predict_age(self, rgb_image: Image.Image) -> float | None: ...


class InsightFaceAgeEstimator:
    """InsightFace buffalo_l — 공개 얼굴 분석 모델 (나이 추정)."""

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "genderage"],
        )
        self.app.prepare(ctx_id=-1, det_size=(224, 224))

    def predict_age(self, rgb_image: Image.Image) -> float | None:
        import cv2

        bgr = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
        faces = self.app.get(bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
        age = getattr(face, "age", None)
        if age is None:
            return None
        age_f = float(age)
        return max(config.AGE_MIN, min(config.AGE_MAX, age_f))


# OpenCV Caffe age net — InsightFace 없이 동작 (learnopencv proto + mirror weights)
OPENCV_AGE_PROTO_URL = (
    "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt"
)
OPENCV_AGE_MODEL_URLS = (
    "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel",
    "https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/"
    "5b60d9f8a8608cdbbcdaaa39bf28f351e8d8553b/model/age_net.caffemodel",
    "https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=1",
)
OPENCV_AGE_MIN_BYTES = 40_000_000  # ~44MB; smaller = failed/partial download
OPENCV_AGE_BUCKET_MIDPOINTS = (1.0, 5.0, 10.0, 17.5, 28.5, 40.5, 50.5, 80.0)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)


def ensure_opencv_age_weights(weights: Path, proto: Path) -> None:
    if not proto.is_file():
        print(f"Downloading {proto.name} ...")
        download_file(OPENCV_AGE_PROTO_URL, proto)

    if weights.is_file() and weights.stat().st_size >= OPENCV_AGE_MIN_BYTES:
        return

    if weights.is_file():
        weights.unlink()

    last_err: Exception | None = None
    for url in OPENCV_AGE_MODEL_URLS:
        try:
            print(f"Downloading {weights.name} from ...{url[-48:]}")
            download_file(url, weights)
            if weights.stat().st_size >= OPENCV_AGE_MIN_BYTES:
                print(f"  OK ({weights.stat().st_size // (1024 * 1024)} MB)")
                return
            weights.unlink()
        except Exception as exc:
            last_err = exc
            print(f"  failed: {exc}")
            if weights.is_file():
                weights.unlink()

    raise RuntimeError(
        "age_net.caffemodel download failed. Manual:\n"
        "  1) Download age_net.caffemodel (~44MB) from\n"
        "     https://github.com/GilLevi/AgeGenderDeepLearning/tree/master/models\n"
        f"  2) Save to: {weights}"
    ) from last_err


class OpenCVAgeEstimator:
    """OpenCV DNN age_net — numpy/skimage 충돌 시 fallback."""

    def __init__(self, model_dir: Path | None = None) -> None:
        import cv2

        model_dir = model_dir or (config.WEIGHTS_DIR / "opencv_age")
        proto = model_dir / "age_deploy.prototxt"
        weights = model_dir / "age_net.caffemodel"
        ensure_opencv_age_weights(weights, proto)

        self.net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        self.mean = (78.4263377603, 87.7689143744, 114.895847746)

    def predict_age(self, rgb_image: Image.Image) -> float | None:
        import cv2

        bgr = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        if h < 8 or w < 8:
            return None
        blob = cv2.dnn.blobFromImage(
            bgr, scalefactor=1.0, size=(227, 227), mean=self.mean, swapRB=False
        )
        self.net.setInput(blob)
        preds = self.net.forward()[0]
        idx = int(np.argmax(preds))
        if idx < 0 or idx >= len(OPENCV_AGE_BUCKET_MIDPOINTS):
            return None
        age_f = float(OPENCV_AGE_BUCKET_MIDPOINTS[idx])
        return max(config.AGE_MIN, min(config.AGE_MAX, age_f))


def create_age_estimator(backend: str) -> AgeEstimator:
    backend = backend.lower()
    if backend == "opencv":
        print("Age backend: OpenCV DNN (age_net)")
        return OpenCVAgeEstimator()

    if backend == "insightface":
        print("Age backend: InsightFace buffalo_l")
        return InsightFaceAgeEstimator()

    # auto: InsightFace -> OpenCV
    try:
        print("Age backend: trying InsightFace buffalo_l ...")
        return InsightFaceAgeEstimator()
    except Exception as exc:
        print(f"  InsightFace failed ({type(exc).__name__}: {exc})")
        print("  Falling back to OpenCV DNN age_net.")
        print(
            "  To fix InsightFace: pip install --force-reinstall scikit-image insightface"
        )
        return OpenCVAgeEstimator()


def process_split(
    source_root: Path,
    split_in: str,
    split_out: str,
    output_root: Path,
    class_names: list[str],
    age_estimator: AgeEstimator,
    padding: float,
    min_crop: int,
    stats: Stats,
    limit: int | None,
) -> list[dict[str, str | int | float]]:
    img_dir = source_root / split_in / "images"
    lbl_dir = source_root / split_in / "labels"
    if not img_dir.is_dir():
        print(f"  skip {split_in}: images 폴더 없음 ({img_dir})")
        return []

    out_face_dir = output_root / "faces" / split_out
    out_face_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []

    images = sorted(
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if limit is not None:
        images = images[:limit]

    for img_path in images:
        stats.images_seen += 1
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.is_file():
            stats.skipped_no_label += 1
            continue

        label_lines = [
            ln for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        box = pick_best_box(label_lines)
        if box is None:
            stats.skipped_no_box += 1
            continue

        class_id, xc, yc, bw, bh = box
        expr_id = class_id_to_expression_id(class_id, class_names)
        if expr_id is None:
            stats.skipped_unknown_class += 1
            continue

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            x1, y1, x2, y2 = yolo_box_to_pixels(xc, yc, bw, bh, w, h, padding)
            if x2 - x1 < min_crop or y2 - y1 < min_crop:
                stats.skipped_small_crop += 1
                continue
            crop = im.crop((x1, y1, x2, y2))

        age = age_estimator.predict_age(crop)
        if age is None:
            stats.skipped_age_failed += 1
            continue

        out_name = f"{img_path.stem}.jpg"
        rel_path = f"faces/{split_out}/{out_name}"
        out_path = output_root / rel_path
        crop.save(out_path, quality=95)

        rows.append(
            {
                "image_path": rel_path.replace("\\", "/"),
                "expression_id": expr_id,
                "age": round(age, 1),
            }
        )
        stats.saved += 1
        stats.class_counts[expr_id] = stats.class_counts.get(expr_id, 0) + 1

        if stats.saved % 500 == 0:
            print(f"  [{split_out}] saved {stats.saved} ...")

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "expression_id", "age"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def write_readme(output_root: Path, source: Path, class_names: list[str]) -> None:
    mapping_lines = []
    for i, name in enumerate(class_names):
        eid = class_id_to_expression_id(i, class_names)
        expr = config.EXPRESSION_LABELS[eid] if eid is not None else "SKIP"
        mapping_lines.append(f"| {i} | {name} | {eid} | {expr} |")

    text = f"""# MTL 학습 데이터 (표정 + 나이)

## 생성

```powershell
python scripts/prepare_mtl_from_roboflow_emotion.py ^
  --source "{source}"
```

## 원본 클래스 → expression_id

| Roboflow id | name | expression_id | config 라벨 |
|-------------|------|---------------|-------------|
{chr(10).join(mapping_lines)}

## CSV 형식

`image_path,expression_id,age` — `data_root`는 `{output_root.name}/`

나이: InsightFace `buffalo_l` (genderage 모듈) 추정값.
"""
    (output_root / "DATASET.md").write_text(text, encoding="utf-8")


def print_stats(stats: Stats) -> None:
    print("\n=== 요약 ===")
    print(f"  images_seen:          {stats.images_seen}")
    print(f"  saved:                {stats.saved}")
    print(f"  skipped_no_label:     {stats.skipped_no_label}")
    print(f"  skipped_no_box:       {stats.skipped_no_box}")
    print(f"  skipped_small_crop:   {stats.skipped_small_crop}")
    print(f"  skipped_unknown_class:{stats.skipped_unknown_class}")
    print(f"  skipped_age_failed:   {stats.skipped_age_failed}")
    if stats.class_counts:
        print("  per expression_id:")
        for eid in sorted(stats.class_counts):
            label = config.EXPRESSION_LABELS[eid]
            print(f"    {eid} ({label}): {stats.class_counts[eid]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roboflow emotion YOLO → MTL face crops + CSV (expression + age)"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(r"C:\Users\moonjintae\Downloads\emotion detection.v1i.yolov8"),
        help="Roboflow YOLOv8 export root (data.yaml 포함)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.MTL_DATA_DIR,
        help="출력 디렉터리 (기본: data/mtl)",
    )
    parser.add_argument("--padding", type=float, default=0.12, help="bbox 패딩 비율")
    parser.add_argument("--min-crop", type=int, default=32, help="최소 크롭 변 길이(px)")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid"],
        help="처리할 원본 split (기본: train valid)",
    )
    parser.add_argument("--limit", type=int, default=None, help="split당 최대 이미지 수 (테스트용)")
    parser.add_argument(
        "--age-backend",
        choices=("auto", "insightface", "opencv"),
        default="auto",
        help="나이 추정: auto(InsightFace 실패 시 OpenCV), insightface, opencv",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)

    class_names = load_roboflow_names(source)
    print("Roboflow classes:", class_names)
    print("config.EXPRESSION_LABELS:", config.EXPRESSION_LABELS)
    print(f"Source: {source}")
    print(f"Output: {output}\n")

    age_estimator = create_age_estimator(args.age_backend)

    stats = Stats()
    all_rows: dict[str, list] = {}

    for split_in in args.splits:
        split_out = SPLIT_MAP.get(split_in, split_in)
        print(f"\nProcessing {split_in} -> {split_out}")
        rows = process_split(
            source,
            split_in,
            split_out,
            output,
            class_names,
            age_estimator,
            args.padding,
            args.min_crop,
            stats,
            args.limit,
        )
        all_rows[split_out] = rows

    if all_rows.get("train"):
        write_csv(output / "train.csv", all_rows["train"])
    if all_rows.get("val"):
        write_csv(output / "val.csv", all_rows["val"])

    write_readme(output, source, class_names)
    print_stats(stats)

    if stats.saved == 0:
        raise SystemExit("저장된 샘플 0개 — 경로·나이 모델(--age-backend)을 확인하세요.")
    print("\n다음: notebooks/mtl/mtl_train_and_tune.ipynb 또는 train/train_mtl.py")


if __name__ == "__main__":
    main()
