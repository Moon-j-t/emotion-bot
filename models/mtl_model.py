"""MobileNetV3 공유 백본 + 표정/나이 Multi-Task Learning 헤드."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
)


class FaceAttributeMTL(nn.Module):
    """
    단일 forward pass로 표정(분류)과 나이(회귀)를 동시에 예측.
    백본 연산을 공유하여 두 개의 독립 모델 대비 연산량을 절감한다.
    """

    def __init__(
        self,
        num_expressions: int = 7,
        variant: str = "large",
        pretrained_backbone: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_expressions = num_expressions
        variant = variant.lower()

        if variant == "small":
            weights = (
                MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            )
            backbone = models.mobilenet_v3_small(weights=weights)
            feat_dim = 576
        elif variant == "large":
            weights = (
                MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            )
            backbone = models.mobilenet_v3_large(weights=weights)
            feat_dim = 960
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        self.expression_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_expressions),
        )
        self.age_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, 1),
        )

        self._init_heads()

    def _init_heads(self) -> None:
        for module in (self.expression_head, self.age_head):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        expression_logits = self.expression_head(features)
        age = self.age_head(features).squeeze(-1)
        return expression_logits, age

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        expr_logits, age = self.forward(x)
        expr_probs = torch.softmax(expr_logits, dim=-1)
        return {
            "expression_logits": expr_logits,
            "expression_probs": expr_probs,
            "age": age,
        }
