import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class GolfKeypointDataset(Dataset):
    def __init__(self, root_dir, annotation_file, sequence_length=16, transform=None):
        """
        root_dir: directory with subfolders of video frames.
        annotation_file: JSON file with {"video_id": [[x1,y1,x2,y2,...], ...]} format.
        sequence_length: number of frames per sample.
        transform: torchvision transform to apply to images.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform

        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        self.samples = []
        for video_id, keypoint_list in self.annotations.items():
            num_frames = len(keypoint_list)
            for i in range(num_frames - sequence_length + 1):
                self.samples.append((video_id, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, start_idx = self.samples[idx]
        frames = []
        keypoints = []

        for i in range(self.sequence_length):
            frame_idx = start_idx + i
            frame_path = os.path.join(self.root_dir, video_id, f"frame_{frame_idx:04d}.jpg")
            kp = np.array(self.annotations[video_id][frame_idx])  # (K*2,)
            img = Image.open(frame_path).convert('RGB')

            if self.transform:
                img = self.transform(img)

            frames.append(img)
            keypoints.append(kp.reshape(-1, 2))  # (K, 2)

        frames = torch.stack(frames, dim=0)  # (T, C, H, W)
        keypoints = torch.tensor(np.stack(keypoints), dtype=torch.float32)  # (T, K, 2)

        return frames, keypoints

# Example transform and usage
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# dataset = GolfKeypointDataset("/path/to/data", "/path/to/annotations.json", transform=transform)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
