# models/hybrid_model.py — Combines CNN and TCN
import torch.nn as nn
from models.cnn_backbone import CNNBackbone
from models.tcn_head import TCNHead

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackbone()
        self.temporal = TCNHead()

    def forward(self, x):
        spatial_features = self.backbone(x)
        out = self.temporal(spatial_features)
        return out
