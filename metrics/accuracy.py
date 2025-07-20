import torch

def compute_mae(pred, gt):
    """
    Computes mean absolute error per keypoint.
    pred, gt: (B, T, K, 2)
    """
    return torch.mean(torch.norm(pred - gt, dim=-1)).item()

def compute_pck(pred, gt, threshold=0.05, norm_factor=224):
    """
    Computes Percentage of Correct Keypoints (PCK).
    threshold: normalized distance (e.g., 0.05 of image width)
    """
    dists = torch.norm(pred - gt, dim=-1)  # (B, T, K)
    correct = (dists < (threshold * norm_factor)).float()
    return correct.mean().item()
