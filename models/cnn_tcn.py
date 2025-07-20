import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, output_dim=512):
        super(FeatureExtractor, self).__init__()
        assert backbone in ['resnet18', 'resnet34', 'resnet50']

        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
        else:
            resnet = models.resnet50(pretrained=pretrained)

        layers = list(resnet.children())[:-2]  # Remove avgpool and fc
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.proj(x)
        return x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.25):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                              padding=(kernel_size-1) * dilation_size, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class KeypointRegressionHead(nn.Module):
    def __init__(self, input_channels, num_keypoints):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, num_keypoints * 2, kernel_size=1)
        )
        self.num_keypoints = num_keypoints

    def forward(self, x):
        # x: (B, C, T) -> output: (B, T, K, 2)
        x = self.head(x)  # (B, 2*K, T)
        B, _, T = x.shape
        x = x.view(B, self.num_keypoints, 2, T).permute(0, 3, 1, 2)  # (B, T, K, 2)
        return x

class CNNTCNKeypointModel(nn.Module):
    def __init__(self, cnn_backbone='resnet18', cnn_out_dim=512, tcn_channels=[256, 128], num_keypoints=10):
        super().__init__()
        self.feature_extractor = FeatureExtractor(backbone=cnn_backbone, output_dim=cnn_out_dim)
        self.temporal_model = TemporalConvNet(input_size=cnn_out_dim, num_channels=tcn_channels)
        self.keypoint_head = KeypointRegressionHead(input_channels=tcn_channels[-1], num_keypoints=num_keypoints)

    def forward(self, x_seq):
        # x_seq: (B, T, C, H, W)
        B, T, C, H, W = x_seq.shape
        features = []
        for t in range(T):
            f = self.feature_extractor(x_seq[:, t])  # (B, F)
            features.append(f)
        x = torch.stack(features, dim=2)  # (B, F, T)
        x = self.temporal_model(x)       # (B, C, T)
        keypoints = self.keypoint_head(x)  # (B, T, K, 2)
        return keypoints
