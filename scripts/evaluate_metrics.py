# evaluate_metrics.py — Run post-training evaluation
from utils.metrics import compute_metrics

def evaluate(predictions, ground_truths):
    compute_metrics(predictions, ground_truths)
