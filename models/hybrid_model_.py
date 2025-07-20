import torch
import torch.nn as nn
from .cnn_backbone import CNNBackbone
from .tcn import TemporalConvNet

class KeypointRegressor(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, num_keypoints * 2, kernel_size=1)
        )
        self.num_keypoints = num_keypoints

    def forward(self, x):
        # Input: (B, C, T), Output: (B, T, K, 2)
        x = self.head(x)  # (B, K*2, T)
        B, _, T = x.size()
        x = x.view(B, self.num_keypoints, 2, T).permute(0, 3, 1, 2)  # (B, T, K, 2)
        return x

class CNNTCNKeypointModel(nn.Module):
    def __init__(self, cnn_output_dim=512, tcn_channels=[256, 128], num_keypoints=10):
        super().__init__()
        self.cnn = CNNBackbone(output_dim=cnn_output_dim)
        self.tcn = TemporalConvNet(cnn_output_dim, tcn_channels)
        self.regressor = KeypointRegressor(in_channels=tcn_channels[-1], num_keypoints=num_keypoints)

    def forward(self, x_seq):
        # x_seq: (B, T, C, H, W)
        B, T, C, H, W = x_seq.size()
        feats = [self.cnn(x_seq[:, t]) for t in range(T)]  # list of (B, F)
        x = torch.stack(feats, dim=2)  # (B, F, T)
        x = self.tcn(x)  # (B, C_out, T)
        out = self.regressor(x)  # (B, T, K, 2)
        return out
