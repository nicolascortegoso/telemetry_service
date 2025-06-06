import torch
import pickle
import json
import pandas as pd
import numpy as np
from typing import List

from src.core.models.model import LSTMAutoencoder
from src.core.utils import find_true_sublists, construct_time_intervals


class AnomalyDetector:
    """
    A class for detecting anomalies in time-series data using an LSTM Autoencoder model.
    """

    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int,
                 num_layers: int,
                 model_weights_path: str,
                 scaler_path: str,
                 threshold_path: str,
                 threshold_coefficient: float,
                 window_size: int,
                 device: str = 'cpu'
        ):
        """
        Initializes the AnomalyDetector with an LSTM Autoencoder model and configurations.

        Loads model parameters, scaler, and threshold from environment variables and files.
        """

        self.device = device
        self.model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        # A coefficient to tweak the threshold value obtained during training
        self.threshold_coefficient = threshold_coefficient

        # Window size to pad predictions
        self.window_size = window_size

        # Load scaler 
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        #Load threshold
        with open(threshold_path, 'r') as f:
            self.threshold = json.load(f)['threshold'] * self.threshold_coefficient

    def preprocess(self, df: pd.DataFrame, window_size: int) -> torch.Tensor:
        """
        Preprocesses a DataFrame to create sequences of scaled features for the LSTM model.

        Extracts specified features, scales them using the loaded scaler, and creates sliding window
        sequences of the specified size.

        Args:
            df (pd.DataFrame): Input DataFrame containing columns 'wheel_rpm', 'speed', and 'distance'.
            window_size (int): The size of the sliding window for creating sequences.

        Returns:
            torch.Tensor: A tensor of shape (n_sequences, window_size, n_features) containing the
                scaled feature sequences.
        """

        features = df[['wheel_rpm', 'speed', 'distance']].values
        features_scaled = self.scaler.transform(features)
        sequences = []
        for i in range(len(features_scaled) - window_size + 1):
            sequences.append(features_scaled[i:i + window_size])
        sequences_array = np.array(sequences)  # Combine list of arrays into one numpy array
        return torch.FloatTensor(sequences_array)

    def detect(self, sequences: torch.Tensor) -> List[bool]:
        """
        Detects anomalies in the input sequences using the LSTM Autoencoder model.

        Computes the reconstruction error for each sequence and compares it to the threshold.

        Args:
            sequences (torch.Tensor): A tensor of shape (n_sequences, window_size, n_features)
                containing the input sequences.

        Returns:
            List[bool]: A list of boolean values indicating whether each sequence is an anomaly
                (True if error > threshold, False otherwise).
        """

        errors = []
        with torch.no_grad():
            for seq in sequences:
                seq = seq.unsqueeze(0).to(self.device)
                output = self.model(seq)
                error = torch.mean((output - seq) ** 2).item()
                errors.append(error)
        return [e > self.threshold for e in errors]

    def anomaly_intervals(self, input_file: str) -> List:
        """
        Identifies time intervals of anomalies in a CSV file using the LSTM Autoencoder model.

        Reads the input file, preprocesses the data, detects anomalies, and returns the timestamp
        intervals corresponding to consecutive anomaly sequences.
        """

        df = pd.read_csv(input_file)
        timestamps = df.pop('timestamp')
        sequences = self.preprocess(df, self.window_size)
        predictions = self.detect(sequences)
        intervals = find_true_sublists(predictions)
        timestamps_intervals = construct_time_intervals(timestamps, intervals)
        return timestamps_intervals
    
    def process_csv_file(self, input_file: str) -> pd.DataFrame:
        """
        Processes a CSV file to detect anomalies and returns a Pandas DataFrame.

        Adds a 'pred' column to the input DataFrame with boolean anomaly predictions and saves
        the modified DataFrame to the output file.
        """

        df = pd.read_csv(input_file)
        n_rows = df.shape[0]
        sequences = self.preprocess(df, self.window_size)
        predictions = self.detect(sequences)
        padded_predictions = [False] * (self.window_size - 1) + predictions
        if len(padded_predictions) > n_rows:
            padded_predictions = padded_predictions[:n_rows]
        elif len(padded_predictions) < n_rows:
            padded_predictions += [False] * (n_rows - len(padded_predictions))
        df['pred'] = padded_predictions
        return df        
    
    def create_file(self, input_file: str, output_file: str) -> None:
        """
        Saves the Pandas DataFrame to results to a new CSV file.
        """

        labeled_dataframe = self.process_csv_file(input_file)
        labeled_dataframe.to_csv(output_file, index=False)
