import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


class ResNet18Classifier(nn.Module):
    """
    ResNet18 backbone adapted for 13-target multi-label pathology classification.
    Outputs raw unbounded logits (no internal sigmoid).
    """

    def __init__(
        self,
        num_classes: int = 13,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
