import torch
import os

from training.preprocess import preprocess_data
from training.model import train_model
from training.evaluate import evaluate_model


def run_experiments(
    train_file: str,
    val_file: str,
    test_file: str,
    window_sizes: list,
    step_sizes: list,
    input_size: int,
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    log_dir: str,
    model_save_path: str,
    threshold_percentile: float
):
    """
    Runs experiments over window sizes and step sizes, saves the best model with threshold,
    and logs results to TensorBoard.

    Args:
        train_file (str): Path to training CSV.
        val_file (str): Path to validation CSV.
        test_file (str): Path to test CSV.
        window_sizes (list): List of window sizes to test.
        step_sizes (list): List of step sizes to test.
        input_size (int): 
        hidden_dim (int): Number of hidden units in LSTM.
        num_layers (int): Number of LSTM layers.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        log_dir (str): TensorBoard log directory.
        model_save_path (str): Path to save the best model.
        threshold_percentile (float): Percentile for anomaly threshold (default: 99.95).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_f1 = 0.0
    best_model_state = None
    best_params = None
    best_threshold = None

    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for window_size in window_sizes:
        for step_size in step_sizes:
            if step_size > window_size:
                continue
            print(f"Running: window_size={window_size}, step_size={step_size}")

            # Preprocess data
            train_sequences, val_sequences, test_sequences, scaler, test_labels = preprocess_data(
                train_file, val_file, test_file, window_size, step_size
            )

            # Train model
            result = train_model(
                train_sequences, val_sequences, window_size, step_size,
                input_size, hidden_dim, num_layers, epochs, batch_size, log_dir, device
            )
            model, history, log_dir_run = result['model'], result['history'], result['log_dir']

            # Evaluate model
            eval_result = evaluate_model(
                model, val_sequences, test_sequences, test_labels, log_dir_run, epochs, device, threshold_percentile
            )
            metrics, run_best_f1, threshold = eval_result['metrics'], eval_result['best_f1'], eval_result['threshold']

            # Print results
            for thresh, m in metrics.items():
                print(f"Threshold {thresh}: Precision={m['precision']:.4f}, "
                      f"Recall={m['recall']:.4f}, F1={m['f1']:.4f}")

            # Save best model
            if run_best_f1 > best_f1:
                best_f1 = run_best_f1
                best_model_state = model.state_dict()
                best_params = {
                    'window_size': window_size,
                    'step_size': step_size,
                    'input_size': input_size,
                    'hidden_dim': hidden_dim
                }
                best_threshold = threshold
                torch.save({
                    'model_state_dict': best_model_state,
                    'params': best_params,
                    'scaler': scaler,
                    'threshold': best_threshold
                }, model_save_path)
                print(f"New best model saved with F1-score: {best_f1:.4f} and threshold: {best_threshold:.4f} at {model_save_path}")

    print(f"Best F1-score: {best_f1:.4f} with params: {best_params}, threshold: {best_threshold:.4f}")