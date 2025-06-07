import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def preprocess_data(
    train_file: str,
    val_file: str,
    test_file: str,
    window_size: int,
    step_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, np.ndarray]:
    """
    Reads training, validation, and test CSV files, normalizes features using training data,
    and returns windowed sequences and test labels.

    Args:
        train_file (str): Path to training CSV file (unlabeled).
        val_file (str): Path to validation CSV file (unlabeled).
        test_file (str): Path to test CSV file (labeled).
        window_size (int): Number of time steps in each sequence window. Default is 15.
        step_size (int, optional): Number of time steps to slide the window. If None, defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, np.ndarray]: Training sequences,
            validation sequences, test sequences, fitted scaler, and test sequence labels.

    Raises:
        ValueError: If step_size is <= 0 or > window_size.
    """
    # Read data
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    train_features = train_df[['wheel_rpm', 'speed', 'distance']].values
    val_features = val_df[['wheel_rpm', 'speed', 'distance']].values
    test_features = test_df[['wheel_rpm', 'speed', 'distance']].values

    # Extract test labels
    test_labels = test_df['is_anomaly'].values

    # Fit scaler on training data
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)

    # Set step_size
    if step_size <= 0 or step_size > window_size:
        raise ValueError(f"step_size must be between 1 and window_size ({window_size}), got {step_size}")

    # Generate sequences
    train_sequences = []
    for i in range(0, len(train_features_scaled) - window_size + 1, step_size):
        train_sequences.append(train_features_scaled[i:i + window_size])
    train_sequences = np.array(train_sequences)

    val_sequences = []
    for i in range(0, len(val_features_scaled) - window_size + 1, step_size):
        val_sequences.append(val_features_scaled[i:i + window_size])
    val_sequences = np.array(val_sequences)

    test_sequences = []
    test_sequence_labels = []
    for i in range(0, len(test_features_scaled) - window_size + 1, step_size):
        test_sequences.append(test_features_scaled[i:i + window_size])
        window_labels = test_labels[i:i + window_size]
        test_sequence_labels.append(1 if np.any(window_labels == 1) else 0)
    test_sequences = np.array(test_sequences)
    test_sequence_labels = np.array(test_sequence_labels)

    return train_sequences, val_sequences, test_sequences, scaler, test_sequence_labels
