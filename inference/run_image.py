"""이미지/동영상 파일 추론."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.run_webcam import draw_predictions  # noqa: E402
from models.pipeline import FaceAnalysisPipeline  # noqa: E402


def predictions_to_dict(predictions):
    return [
        {
            "track_id": p.track_id,
            "box": list(p.box),
            "detection_conf": p.detection_conf,
            "expression": p.expression,
            "expression_confidence": p.expression_confidence,
            "expression_probs": p.expression_probs.tolist(),
            "age": p.age,
            "raw_age": p.raw_age,
            "smooth_frames": p.smooth_frames,
        }
        for p in predictions
    ]


def run_image(pipeline, image_path: Path, output_path: Path | None) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(image_path)

    preds = pipeline.process_frame(frame)
    frame = draw_predictions(frame, preds)
    print(json.dumps(predictions_to_dict(preds), ensure_ascii=False, indent=2))

    if output_path:
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")


def run_video(pipeline, video_path: Path, output_path: Path | None) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    writer = None
    if output_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        preds = pipeline.process_frame(frame)
        frame = draw_predictions(frame, preds)
        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
        print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Image or video path")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--yolo", type=str, default=None)
    parser.add_argument("--mtl", type=str, default=None)
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    pipeline = FaceAnalysisPipeline(
        yolo_model=args.yolo,
        mtl_weights=args.mtl,
        device=args.device,
        smooth_window=args.smooth_window,
    )

    suffix = args.input.suffix.lower()
    if suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        run_video(pipeline, args.input, args.output)
    else:
        run_image(pipeline, args.input, args.output)


if __name__ == "__main__":
    main()
