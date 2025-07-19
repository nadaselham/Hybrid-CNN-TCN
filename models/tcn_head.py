# tcn_head.py — Temporal modeling block using 1D convolutions
import torch.nn as nn

class TCNHead(nn.Module):
    def __init__(self, input_channels=512, output_channels=13):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(input_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, output_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # reshape from (N, C, H, W) to (N, C, T)
        x = x.view(x.size(0), x.size(1), -1)
        return self.temporal(x)
