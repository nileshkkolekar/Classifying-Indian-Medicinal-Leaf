"""The classifier: a pretrained backbone plus a small linear head.

With a few hundred images per species, training from scratch is not viable —
transfer learning from ImageNet is what makes this problem tractable. The
backbone comes from ``timm`` so any of its several hundred architectures can
be swapped in by name from the config.
"""

from __future__ import annotations

import logging
from typing import cast

import timm
import torch
from torch import nn

logger = logging.getLogger(__name__)


class LeafClassifier(nn.Module):
    """Feature extractor + dropout + linear head.

    ``timm.create_model(..., num_classes=0)`` returns pooled features rather
    than logits, so the head stays explicit and easy to reset or re-shape.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 5,
        *,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")

        self.backbone_name = backbone
        self.num_classes = num_classes

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        # nn.Module.__getattr__ is typed as returning Tensor | Module, so the
        # backbone's feature count needs a cast to stay honest about being int.
        self.num_features: int = cast(int, self.backbone.num_features)

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_features, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

        logger.info(
            "Built %s (%d features -> %d classes, pretrained=%s)",
            backbone,
            self.num_features,
            num_classes,
            pretrained,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled embeddings — useful for similarity search and error analysis."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def freeze_backbone(self) -> None:
        """Train the head only — a fast, low-variance first baseline."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        logger.info("Backbone frozen; %d trainable parameters remain", self.trainable_parameters)

    def unfreeze_backbone(self) -> None:
        """Re-enable full fine-tuning, usually with a reduced learning rate."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train(self.training)
        logger.info("Backbone unfrozen; %d trainable parameters", self.trainable_parameters)

    @property
    def backbone_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.backbone.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train(self, mode: bool = True) -> LeafClassifier:
        """Keep a frozen backbone in eval mode so its BatchNorm stats hold still."""
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities; sets eval mode for the duration."""
        was_training = self.training
        self.eval()
        try:
            return torch.softmax(self(x), dim=1)
        finally:
            self.train(was_training)

    def extra_repr(self) -> str:
        return f"backbone={self.backbone_name}, num_classes={self.num_classes}"
