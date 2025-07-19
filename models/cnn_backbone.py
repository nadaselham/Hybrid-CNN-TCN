import torch.nn as nn
import torchvision.models as models

class CNNBackbone(nn.Module):
    def __init__(self, pretrained=True, output_dim=512):
        super(CNNBackbone, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x
