"""YOLO 검출 → MobileNetV3 MTL → Temporal Smoothing 통합 파이프라인."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

import config
from models.detector import FaceDetector
from models.mtl_model import FaceAttributeMTL
from models.preprocess import build_transform, faces_to_batch
from smoothing.temporal_smoother import TemporalSmoother
from smoothing.tracker import SimpleIoUTracker, TrackedFace


@dataclass
class FacePrediction:
    track_id: int
    box: tuple[int, int, int, int]
    detection_conf: float
    expression: str
    expression_confidence: float
    expression_probs: np.ndarray
    age: float
    raw_age: float
    smooth_frames: int


class FaceAnalysisPipeline:
    def __init__(
        self,
        yolo_model: str | None = None,
        mtl_weights: Path | str | None = None,
        device: str | None = None,
        smooth_window: int | None = None,
    ) -> None:
        self.device = torch.device(
            device
            or config.DEVICE
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.detector = FaceDetector(
            model_path=yolo_model or config.YOLO_MODEL,
            conf=config.YOLO_CONF,
            iou=config.YOLO_IOU,
            imgsz=config.YOLO_IMGSZ,
            device=str(self.device),
        )

        self.mtl = FaceAttributeMTL(
            num_expressions=config.NUM_EXPRESSIONS,
            variant=config.MOBILENET_VARIANT,
            pretrained_backbone=True,
        ).to(self.device)
        self.mtl.eval()

        weights_path = Path(mtl_weights or config.MTL_WEIGHTS)
        if weights_path.is_file():
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.mtl.load_state_dict(state)
        else:
            # 데모: ImageNet 백본 + 랜덤 헤드. 실사용 시 train/train_mtl.py로 학습 권장.
            pass

        self.transform = build_transform(config.INPUT_SIZE)
        self.tracker = SimpleIoUTracker(
            iou_threshold=config.TRACK_IOU_THRESHOLD,
            max_missed=config.TRACK_MAX_MISSED,
        )
        self.smoother = TemporalSmoother(
            window_size=smooth_window or config.SMOOTH_WINDOW,
            num_expressions=config.NUM_EXPRESSIONS,
        )

    @torch.inference_mode()
    def process_frame(self, frame_bgr: np.ndarray) -> list[FacePrediction]:
        detections = self.detector.detect(frame_bgr)
        tracked: list[TrackedFace] = self.tracker.update(detections)
        if not tracked:
            self.smoother.prune(set())
            return []

        active_ids = {t.track_id for t in tracked}
        self.smoother.prune(active_ids)

        boxes = [t.face.xyxy for t in tracked]
        batch = faces_to_batch(frame_bgr, boxes, self.transform, self.device)
        if batch is None:
            return []

        expr_logits, ages = self.mtl(batch)
        expr_logits_np = expr_logits.cpu().numpy()
        ages_np = ages.cpu().numpy()

        predictions: list[FacePrediction] = []
        for i, tracked_face in enumerate(tracked):
            raw_age = float(ages_np[i])
            clamped_age = float(
                np.clip(raw_age, config.AGE_MIN, config.AGE_MAX)
            )

            smooth_probs, smooth_age, n_frames = self.smoother.update(
                track_id=tracked_face.track_id,
                expression_logits=expr_logits_np[i],
                age=clamped_age,
            )

            expr_idx = int(np.argmax(smooth_probs))
            predictions.append(
                FacePrediction(
                    track_id=tracked_face.track_id,
                    box=tracked_face.face.xyxy,
                    detection_conf=tracked_face.face.confidence,
                    expression=config.EXPRESSION_LABELS[expr_idx],
                    expression_confidence=float(smooth_probs[expr_idx]),
                    expression_probs=smooth_probs,
                    age=smooth_age,
                    raw_age=clamped_age,
                    smooth_frames=n_frames,
                )
            )
        return predictions
