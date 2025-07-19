# cnn_backbone.py — Example ResNet-based feature extractor
import torchvision.models as models
import torch.nn as nn

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.features(x)
