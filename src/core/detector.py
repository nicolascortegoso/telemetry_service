import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import label

from src.core.models.model import LSTMAutoencoder


class AnomalyDetector:
    def __init__(
        self,
        model_path: str,
        batch_size: int,
        device: str,
    ):
        """
        Initializes the anomaly detector with the saved model and threshold.

        Args:
            model_path (str): Path to saved model.
            batch_size (int): Batch size for inference.
            device (str): Device ('cuda', 'mps' or 'cpu').
        """

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.params = checkpoint['params']
        self.window_size = self.params['window_size']
        self.step_size = self.params['step_size']
        self.input_size = self.params['input_size']
        self.hidden_dim = self.params['hidden_dim']
        self.scaler = checkpoint['scaler']
        self.threshold = checkpoint['threshold']

        # Initialize model
        self.model = LSTMAutoencoder(input_dim=self.input_size, hidden_dim=self.hidden_dim, num_layers=1).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Set inference parameters
        self.batch_size = batch_size
        self.device = device

    def inference(
        self,
        data_file: str,
        output_dir: str = "logs/inference"
    ) -> tuple:
        """
        Runs inference on new data, detects anomalies, and saves predictions to CSV.

        Args:
            data_file (str): Path to new data CSV (with 'wheel_rpm', 'speed', 'distance').
            output_dir (str): Directory to save predictions CSV.

        Returns:
            tuple: Sequence-level predictions, time step-level predictions, and MAE scores.
        """

        # Preprocess new data
        new_df = pd.read_csv(data_file)
        if 'timestamp' not in new_df.columns:
            raise ValueError("Input CSV must contain a 'timestamp' column")
        timestamps = pd.to_datetime(new_df['timestamp'])
        new_features = new_df[['wheel_rpm', 'speed', 'distance']].values
        new_features_scaled = self.scaler.transform(new_features)

        # Generate sequences
        new_sequences = []
        for i in range(0, len(new_features_scaled) - self.window_size + 1, self.step_size):
            new_sequences.append(new_features_scaled[i:i + self.window_size])
        new_sequences = np.array(new_sequences)
        new_tensor = torch.tensor(new_sequences, dtype=torch.float32).to(self.device)
        new_loader = DataLoader(TensorDataset(new_tensor, new_tensor), batch_size=self.batch_size, shuffle=False)

        # Run inference
        sequence_mae = []
        with torch.no_grad():
            for batch_x, _ in new_loader:
                output = self.model(batch_x)
                mae = torch.mean(torch.abs(output - batch_x), dim=(1, 2)).cpu().numpy()
                sequence_mae.extend(mae)
        sequence_mae = np.array(sequence_mae)

        # Detect anomalies
        sequence_predictions = (sequence_mae > self.threshold).astype(int)

        # Map sequence predictions to time steps
        time_step_predictions = np.zeros(len(new_features))
        for i, idx in enumerate(range(0, len(new_features_scaled) - self.window_size + 1, self.step_size)):
            if sequence_predictions[i] == 1:
                time_step_predictions[idx:idx + self.window_size] = 1

        # Identify anomaly intervals
        labeled_array, num_features = label(time_step_predictions)
        anomaly_intervals = []
        for i in range(1, num_features + 1):
            segment = (labeled_array == i)
            indices = np.where(segment)[0]
            if len(indices) > 0:
                start_ts = timestamps[indices[0]].strftime('%Y-%m-%d %H:%M:%S')
                end_ts = timestamps[indices[-1]].strftime('%Y-%m-%d %H:%M:%S')
                anomaly_intervals.append({"start": start_ts, "end": end_ts})

        output_df = new_df.copy()
        output_df['predicted'] = time_step_predictions.astype(bool)

        return sequence_predictions, time_step_predictions, sequence_mae, output_df, anomaly_intervals