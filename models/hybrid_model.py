import torch.nn as nn
from .cnn_backbone import CNNBackbone
from .tcn_module import TemporalConvNet

class CNNTCNKeypointModel(nn.Module):
    def __init__(self, cnn_output_dim=512, tcn_channels=[256, 128], num_keypoints=10):
        super(CNNTCNKeypointModel, self).__init__()
        self.cnn = CNNBackbone(output_dim=cnn_output_dim)
        self.tcn = TemporalConvNet(input_size=cnn_output_dim, num_channels=tcn_channels)
        self.keypoint_head = nn.Conv1d(tcn_channels[-1], num_keypoints * 2, kernel_size=1)

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.size()
        cnn_features = []
        for t in range(T):
            feat = self.cnn(x_seq[:, t])
            cnn_features.append(feat)
        x = torch.stack(cnn_features, dim=2)  # (B, F, T)
        x = self.tcn(x)
        x = self.keypoint_head(x)  # (B, 2*num_kp, T)
        x = x.view(B, -1, 2, T).permute(0, 3, 1, 2)  # (B, T, num_kp, 2)
        return x
