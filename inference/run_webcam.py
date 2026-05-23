"""웹캠 실시간: YOLO + MobileNetV3 MTL + 프레임 스무딩."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.pipeline import FaceAnalysisPipeline  # noqa: E402


def draw_predictions(frame, predictions):
    for pred in predictions:
        x1, y1, x2, y2 = pred.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 120), 2)
        label = (
            f"id={pred.track_id} {pred.expression} "
            f"({pred.expression_confidence:.2f}) age={pred.age:.1f} "
            f"[{pred.smooth_frames}f]"
        )
        cv2.putText(
            frame,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 120),
            2,
            cv2.LINE_AA,
        )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam face attribute analysis")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--yolo",
        type=str,
        default=None,
        help="Trained YOLO weights (default: weights/face_yolo.pt)",
    )
    parser.add_argument("--mtl", type=str, default=None, help="MTL checkpoint path")
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    pipeline = FaceAnalysisPipeline(
        yolo_model=args.yolo,
        mtl_weights=args.mtl,
        device=args.device,
        smooth_window=args.smooth_window,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        predictions = pipeline.process_frame(frame)
        frame = draw_predictions(frame, predictions)
        cv2.imshow("Face MTL (YOLO + MobileNetV3)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
