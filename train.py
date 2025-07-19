# train.py — Training script for hybrid CNN-TCN keypoint detector
import torch
from models.hybrid_model import HybridModel
from data.transforms import get_transforms
from utils.metrics import compute_metrics
from utils.logger import Logger
import yaml

def train(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = HybridModel()
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    logger = Logger()

    # Load dataset, apply transforms, create dataloader...
    # Example placeholder
    for epoch in range(config["epochs"]):
        model.train()
        for batch in train_loader:
            # forward, loss, backward, step
            pass
        logger.log_epoch(epoch)

    torch.save(model.state_dict(), "outputs/best_model.pth")

if __name__ == "__main__":
    train("configs/config.yaml")
