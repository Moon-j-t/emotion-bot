"""직접 학습한 YOLO 얼굴 검출 모델 로더."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

import config


@dataclass
class FaceBox:
    xyxy: tuple[int, int, int, int]
    confidence: float

    @property
    def x1(self) -> int:
        return self.xyxy[0]

    @property
    def y1(self) -> int:
        return self.xyxy[1]

    @property
    def x2(self) -> int:
        return self.xyxy[2]

    @property
    def y2(self) -> int:
        return self.xyxy[3]


def resolve_yolo_weights(model_path: str | Path | None = None) -> Path:
    """학습된 얼굴 YOLO 가중치 경로를 확인한다."""
    path = Path(model_path or config.YOLO_MODEL)
    if path.is_file():
        return path.resolve()

    raise FileNotFoundError(
        f"학습된 YOLO 얼굴 검출 가중치가 없습니다: {path}\n"
        "1) data/face_detect 에 이미지·라벨을 준비하고\n"
        "2) python train/train_yolo_face.py 로 학습하세요.\n"
        "   (학습 완료 시 weights/face_yolo.pt 가 생성됩니다)"
    )


class FaceDetector:
    def __init__(
        self,
        model_path: str | Path | None = None,
        conf: float = 0.45,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str | None = None,
        require_trained: bool = True,
    ) -> None:
        weights = (
            resolve_yolo_weights(model_path)
            if require_trained
            else Path(model_path or config.YOLO_MODEL)
        )
        self.weights_path = weights
        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes: list[FaceBox] = []
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        h, w = frame_bgr.shape[:2]
        for (x1, y1, x2, y2), score in zip(xyxy, confs):
            xi1 = int(max(0, min(w - 1, x1)))
            yi1 = int(max(0, min(h - 1, y1)))
            xi2 = int(max(0, min(w, x2)))
            yi2 = int(max(0, min(h, y2)))
            if xi2 <= xi1 or yi2 <= yi1:
                continue
            boxes.append(
                FaceBox(
                    xyxy=(xi1, yi1, xi2, yi2),
                    confidence=float(score),
                )
            )
        return boxes
