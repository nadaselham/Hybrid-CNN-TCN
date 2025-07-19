# Golf Swing Keypoint Detection (Hybrid CNN-TCN)
Spatiotemporal Key Point Detection in Golf Swing Sequences via Hybrid CNN-TCN Architecture and Regression-Based Refinement


This repository contains the full implementation of our hybrid CNN–TCN model for fine-grained keypoint detection in golf swing sequences. The model integrates spatial and temporal feature modeling to enhance accuracy and anatomical coherence.


Our method extracts spatial features using a CNN backbone (ResNet-18/EfficientNet) and models temporal dependencies using a Temporal Convolutional Network (TCN). This combination is particularly effective for analyzing complex biomechanical motions in the GolfDB dataset.

Summary

- Spatial encoder: CNN (ResNet or EfficientNet)
- Temporal modeling: TCN on feature sequences
- Output: Keypoint regression + heatmap-based supervision
- Loss functions: MSE (heatmap) + MAE (coordinate regression) + Temporal Smoothness

Repository Structure
golfdb-keypoint-hybrid/
│
├── configs/ # YAML config files
├── data/ # Dataset and augmentation scripts
├── models/ # CNN encoder, TCN head, HybridModel
├── scripts/ # Auxiliary tools (e.g., frame extraction)
├── utils/ # Metrics, visualization, logger
│
├── train.py # Training script
├── validate.py # Evaluation script
├── inference.py # Inference on new video
├── requirements.txt # Dependency list
└── README.md # This file

Setup Instructions
1. Clone the repository:
git clone https://github.com/nadaselham/hybrid-cnn-tcn.git
cd hybrid-cnn-tcn
2. Install dependencies:
pip install -r requirements.txt
3. Download GolfDB dataset from official GitHub repo and place it under:
  data/golfdb_dataset/
├── train/
├── val/
5. To train the model:
python train.py --config configs/config.yaml
6. To evaluate a trained model:
python validate.py --checkpoint outputs/best_model.pth



