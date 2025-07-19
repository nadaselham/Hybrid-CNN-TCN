import torch

def compute_mae(pred_kp, true_kp):
    return torch.mean(torch.norm(pred_kp - true_kp, dim=-1))

def compute_pck(pred_kp, true_kp, threshold=0.05):
    distances = torch.norm(pred_kp - true_kp, dim=-1)
    correct = distances < threshold
    return correct.float().mean().item()
