import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import os

from src.core.models.model import LSTMAutoencoder


def train_model(
    train_sequences: np.ndarray,
    val_sequences: np.ndarray,
    window_size: int,
    step_size: int,
    input_size: int,
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    log_dir: str,
    device: str 
) -> dict:
    """
    Trains the LSTM autoencoder, saves the best model based on validation MAE, and logs to TensorBoard.

    Args:
        train_sequences (np.ndarray): Training sequences.
        val_sequences (np.ndarray): Validation sequences.
        window_size (int): Window size used.
        step_size (int): Step size used.
        input_size (int): 
        hidden_dim (int): Number of hidden units in LSTM.
        num_layers (int): Number of LSTM layers.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        log_dir (str): Directory for TensorBoard logs.
        device (str): Device to train on ('cuda' or 'cpu').

    Returns:
        dict: Trained model, training history, and log directory.
    """
    # Create unique log directory
    run_name = f"window_{window_size}_step_{step_size}_hidden_{hidden_dim}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir_run = os.path.join(log_dir, run_name)
    writer = SummaryWriter(log_dir_run)

    # Convert numpy arrays to PyTorch tensors
    train_tensor = torch.tensor(train_sequences, dtype=torch.float32).to(device)
    val_tensor = torch.tensor(val_sequences, dtype=torch.float32).to(device)

    # Create DataLoaders
    train_dataset = TensorDataset(train_tensor, train_tensor)
    val_dataset = TensorDataset(val_tensor, val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = LSTMAutoencoder(input_dim=input_size, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    criterion = nn.L1Loss() # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early stopping parameters
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training history
    history = {'train_loss': [], 'val_loss': []}

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                output = model(batch_x)
                loss = criterion(output, batch_x)
                val_loss += loss.item() * batch_x.size(0)
            val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        print(f'Epoch {epoch + 1} - Loss/train {train_loss} - Loss/val {val_loss}')

        # Early stopping and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save best model for this run
    model.load_state_dict(best_model_state)
    writer.close()

    return {'model': model, 'history': history, 'log_dir': log_dir_run, 'best_val_loss': best_val_loss}