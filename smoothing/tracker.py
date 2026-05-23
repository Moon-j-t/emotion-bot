"""프레임 간 얼굴 박스 IoU 매칭으로 track ID 유지."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.detector import FaceBox


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    box: tuple[int, int, int, int]
    missed: int = 0


@dataclass
class TrackedFace:
    track_id: int
    face: FaceBox


class SimpleIoUTracker:
    def __init__(self, iou_threshold: float = 0.35, max_missed: int = 10) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def update(self, detections: list[FaceBox]) -> list[TrackedFace]:
        if not self._tracks:
            return self._spawn_all(detections)

        track_ids = list(self._tracks.keys())
        track_boxes = [self._tracks[tid].box for tid in track_ids]
        det_boxes = [d.xyxy for d in detections]

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        pairs: list[tuple[int, int, float]] = []

        for ti, tbox in enumerate(track_boxes):
            for di, dbox in enumerate(det_boxes):
                score = _iou(tbox, dbox)
                if score >= self.iou_threshold:
                    pairs.append((ti, di, score))

        pairs.sort(key=lambda x: x[2], reverse=True)
        assignments: dict[int, int] = {}
        for ti, di, _ in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            assignments[track_ids[ti]] = di

        for tid in track_ids:
            if tid not in assignments:
                self._tracks[tid].missed += 1

        output: list[TrackedFace] = []
        for tid, det_idx in assignments.items():
            det = detections[det_idx]
            self._tracks[tid].box = det.xyxy
            self._tracks[tid].missed = 0
            output.append(TrackedFace(track_id=tid, face=det))

        for di, det in enumerate(detections):
            if di not in matched_dets:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _Track(track_id=tid, box=det.xyxy, missed=0)
                output.append(TrackedFace(track_id=tid, face=det))

        stale = [tid for tid, t in self._tracks.items() if t.missed > self.max_missed]
        for tid in stale:
            del self._tracks[tid]

        return output

    def _spawn_all(self, detections: list[FaceBox]) -> list[TrackedFace]:
        output: list[TrackedFace] = []
        for det in detections:
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _Track(track_id=tid, box=det.xyxy)
            output.append(TrackedFace(track_id=tid, face=det))
        return output
