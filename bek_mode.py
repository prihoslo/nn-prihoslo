from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import torch.nn as nn

class MyEfficientNet(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(
            in_features=in_features,
            out_features=num_classes,
            bias=True,
        )

        for param in self.model.parameters():
            param.requires_grad = False

        for idx in (7, 8):
            for param in self.model.features[idx].parameters():
                param.requires_grad = True

        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)