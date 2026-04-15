import os

import torch
from torch import nn
from torchvision.models import (
    resnet18,
    resnet34,
    ResNet18_Weights,
    ResNet34_Weights,
)


class Model(nn.Module):
    def __init__(self, model_name="resnet18", num_classes=10, pretrained=False):
        super().__init__()
        self.weights_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(self.weights_dir, exist_ok=True)
        self.model = self._build_model(model_name=model_name, num_classes=num_classes, pretrained=pretrained)

    def _load_pretrained_state_dict(self, weights_enum):
        return torch.hub.load_state_dict_from_url(
            weights_enum.url,
            model_dir=self.weights_dir,
            progress=True,
            check_hash=True,
            map_location="cpu",
        )

    def _build_model(self, model_name, num_classes, pretrained):
        if model_name == "resnet18":
            model = resnet18(weights=None)
            if pretrained:
                state_dict = self._load_pretrained_state_dict(ResNet18_Weights.DEFAULT)
                model.load_state_dict(state_dict)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model

        if model_name == "resnet34":
            model = resnet34(weights=None)
            if pretrained:
                state_dict = self._load_pretrained_state_dict(ResNet34_Weights.DEFAULT)
                model.load_state_dict(state_dict)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model

        raise ValueError(f"Unsupported model_name: {model_name}")

    def forward(self, x):
        return self.model(x)

