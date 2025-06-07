import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch.utils.tensorboard import SummaryWriter


def evaluate_model(
    model: torch.nn.Module,
    val_sequences: np.ndarray,
    test_sequences: np.ndarray,
    test_labels: np.ndarray,
    log_dir: str,
    epochs: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    threshold_percentile: float = 99.95
) -> dict:
    """
    Evaluates the model on test data, using validation data to set thresholds,
    and logs metrics to TensorBoard.

    Args:
        model: Trained PyTorch model.
        val_sequences (np.ndarray): Validation sequences (unlabeled).
        test_sequences (np.ndarray): Test sequences.
        test_labels (np.ndarray): Sequence-level anomaly labels for test set.
        log_dir (str): TensorBoard log directory.
        epochs (int): Epoch number for logging.
        device (str): Device for inference ('cuda' or 'cpu').
        threshold_percentile (float): Percentile for anomaly threshold (default: 99.95).

    Returns:
        dict: Metrics for each threshold, best F1-score, and validation MAE threshold.
    """
    model.eval()
    writer = SummaryWriter(log_dir)

    # Convert to tensors
    val_tensor = torch.tensor(val_sequences, dtype=torch.float32).to(device)
    test_tensor = torch.tensor(test_sequences, dtype=torch.float32).to(device)

    # Compute reconstruction errors
    with torch.no_grad():
        val_reconstructions = model(val_tensor)
        val_mae = torch.mean(torch.abs(val_reconstructions - val_tensor), dim=(1, 2)).cpu().numpy()
        test_reconstructions = model(test_tensor)
        test_mae = torch.mean(torch.abs(test_reconstructions - test_tensor), dim=(1, 2)).cpu().numpy()

    # Compute validation MAE threshold
    threshold = np.percentile(val_mae, threshold_percentile)

    # Test thresholds based on validation MAE
    thresholds = np.percentile(val_mae, [99.5, 99.9, 99.95])
    metrics = {}
    best_f1 = 0.0
    for i, thresh in enumerate(thresholds):
        predictions = test_mae > thresh
        precision = precision_score(test_labels, predictions, zero_division=0)
        recall = recall_score(test_labels, predictions, zero_division=0)
        f1 = f1_score(test_labels, predictions, zero_division=0)
        metrics[f'thresh_{i}'] = {'precision': precision, 'recall': recall, 'f1': f1}
        best_f1 = max(best_f1, f1)

        # Log metrics to TensorBoard
        writer.add_scalar(f'precision_threshold_{i}', precision, epochs)
        writer.add_scalar(f'recall_threshold_{i}', recall, epochs)
        writer.add_scalar(f'f1_score_threshold_{i}', f1, epochs)

    # Log MAE and threshold
    writer.add_scalar('val_mae', np.mean(val_mae), epochs)
    writer.add_scalar('test_mae', np.mean(test_mae), epochs)
    writer.add_scalar('val_threshold', threshold, epochs)
    writer.close()

    return {'metrics': metrics, 'best_f1': best_f1, 'threshold': threshold}