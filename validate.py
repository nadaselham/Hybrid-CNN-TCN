# validate.py — Validation script to evaluate model performance
import torch
from models.hybrid_model import HybridModel
from utils.metrics import compute_metrics

def validate(model_path, val_loader):
    model = HybridModel()
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.cuda()
            outputs = model(inputs)
            all_preds.append(outputs.cpu())
            all_targets.append(targets)

    compute_metrics(all_preds, all_targets)

if __name__ == "__main__":
    from data.dataloader import get_validation_loader
    val_loader = get_validation_loader()
    validate("outputs/best_model.pth", val_loader)
