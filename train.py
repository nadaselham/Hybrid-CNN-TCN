import torch
from torch.utils.data import DataLoader
from data.dataset import GolfDBKeypointDataset
from models.hybrid_model import CNNTCNHybrid
from experiments.logger import SimpleLogger
from metrics.accuracy import compute_mae
import torch.nn as nn
import torch.optim as optim
import os

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = GolfDBKeypointDataset(config['data_root'], config['annotations'], transform=config['transform'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model = CNNTCNHybrid(num_keypoints=config['num_keypoints']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    logger = SimpleLogger(output_path=config['log_path'])

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0

        for images, keypoints in loader:
            B = images.size(0)
            images = images.view(B, config['seq_len'], 3, 224, 224).to(device)
            keypoints = keypoints.view(B, config['seq_len'], -1, 2).to(device)

            preds = model(images)
            loss = criterion(preds, keypoints)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        mae = compute_mae(preds.detach(), keypoints)
        logger.log(epoch, loss.item(), mae)

        # Save model every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(config['save_dir'], f"model_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    from experiments.configs import get_config
    config = get_config()
    train_model(config)
