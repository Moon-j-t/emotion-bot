"""
YOLO 얼굴 검출 모델 직접 학습 (Ultralytics).

데이터: data/face_detect (data.yaml, images/, labels/)
학습 완료 후 best.pt → weights/face_yolo.pt 로 복사하여 추론에 사용.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def export_best_weights(run_weights_dir: Path, destination: Path) -> Path:
    best_pt = run_weights_dir / "best.pt"
    if not best_pt.is_file():
        raise FileNotFoundError(f"best.pt not found: {best_pt}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO face detector")
    parser.add_argument(
        "--data",
        type=Path,
        default=config.YOLO_DATA_YAML,
        help="data.yaml path",
    )
    parser.add_argument(
        "--base",
        type=str,
        default=config.YOLO_BASE_ARCH,
        help="Base architecture weights for training init (e.g. yolov8n.pt)",
    )
    parser.add_argument("--epochs", type=int, default=config.YOLO_TRAIN_EPOCHS)
    parser.add_argument("--batch", type=int, default=config.YOLO_TRAIN_BATCH)
    parser.add_argument("--imgsz", type=int, default=config.YOLO_IMGSZ)
    parser.add_argument("--patience", type=int, default=config.YOLO_TRAIN_PATIENCE)
    parser.add_argument(
        "--project",
        type=Path,
        default=config.YOLO_TRAIN_PROJECT,
    )
    parser.add_argument("--name", type=str, default=config.YOLO_TRAIN_NAME)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.YOLO_MODEL,
        help="Exported trained weights for inference",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume last training run")
    args = parser.parse_args()

    data_yaml = args.data.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"data.yaml not found: {data_yaml}\n"
            "Run: python scripts/prepare_yolo_dataset.py ..."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("pip install ultralytics") from exc

    model = YOLO(args.base)
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        device=args.device,
        exist_ok=True,
    )
    if args.resume:
        train_kwargs["resume"] = True

    print(f"Training face detector | data={data_yaml} | base={args.base}")
    results = model.train(**train_kwargs)

    save_dir = Path(results.save_dir)
    exported = export_best_weights(save_dir / "weights", args.output.resolve())
    print(f"Exported inference weights: {exported}")
    print(f"Run inference: python inference/run_webcam.py --yolo {exported}")


if __name__ == "__main__":
    main()
