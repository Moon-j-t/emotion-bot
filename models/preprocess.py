"""얼굴 crop 전처리."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def crop_face(frame_bgr: np.ndarray, box: tuple[int, int, int, int], margin: float = 0.12) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * margin)
    pad_y = int(bh * margin)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    return frame_bgr[cy1:cy2, cx1:cx2]


def build_transform(input_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def faces_to_batch(
    frame_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    transform: transforms.Compose,
    device: torch.device,
) -> torch.Tensor | None:
    if not boxes:
        return None

    tensors = []
    for box in boxes:
        crop = crop_face(frame_bgr, box)
        if crop.size == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensors.append(transform(rgb))

    if not tensors:
        return None

    return torch.stack(tensors, dim=0).to(device)
