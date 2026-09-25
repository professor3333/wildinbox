"""The network definition shared by training and serving."""

from __future__ import annotations

from torch import nn


def build_model(num_classes: int, weights: str | None) -> nn.Module:
    """EfficientNet-B0 with a `num_classes` head. `weights` names torchvision
    pretrained weights (training); None builds the bare network (serving)."""
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    model = efficientnet_b0(weights=EfficientNet_B0_Weights[weights] if weights else None)
    head = model.classifier[1]
    assert isinstance(head, nn.Linear)
    model.classifier[1] = nn.Linear(head.in_features, num_classes)
    result: nn.Module = model
    return result
