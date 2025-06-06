import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import os
import pickle
import json
import yaml

from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

from src.core.models.model import LSTMAutoencoder


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# Set random seed for reproducibility
torch.manual_seed(config['training']['random_seed'])
np.random.seed(config['training']['random_seed'])


# Select the best available device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Preprocessing function
def preprocess_data(file_path, window_size=10):
    df = pd.read_csv(file_path)
    features = df[['wheel_rpm', 'speed', 'distance']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    sequences = []
    for i in range(len(features_scaled) - window_size + 1):
        sequences.append(features_scaled[i:i + window_size])
    sequences = np.array(sequences)
    
    return sequences, scaler

# Training function
def train_model(model, train_data, val_data, epochs=50, batch_size=32):
    writer = SummaryWriter('logs/tensorboard')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    writer.close()
    return model

# Compute anomaly threshold
def compute_threshold(model, val_data):
    model.eval()
    errors = []
    with torch.no_grad():
        for seq in val_data:
            seq = torch.FloatTensor(seq).unsqueeze(0).to(device)
            output = model(seq)
            error = torch.mean((output - seq) ** 2, dim=(1, 2)).item()
            errors.append(error)
    threshold = np.mean(errors) + 2 * np.std(errors)
    return threshold

# Plot reconstruction errors with Plotly
def plot_reconstruction_errors(val_data, model, threshold, output_path):
    model.eval()
    errors = []
    with torch.no_grad():
        for seq in val_data:
            seq = torch.FloatTensor(seq).unsqueeze(0).to(device)
            output = model(seq)
            error = torch.mean((output - seq) ** 2, dim=(1, 2)).item()
            errors.append(error)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=errors, mode='lines', name='Reconstruction Error'))
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", name='Threshold')
    fig.update_layout(title='Reconstruction Errors', xaxis_title='Sample', yaxis_title='MSE')
    fig.write_html(output_path)

# Main function
def main():

    window_size = config['model']['window_size']
    hidden_dim = config['model']['hidden_dim']
    num_layers = config['model']['num_layers']
    input_dim = config['model']['input_dim']

    # Paths
    train_data_path = config['paths']['training_data']
    val_data_path = config['paths']['validation_data']
    model_path = config['paths']['model_weights']
    scaler_path = config['paths']['scaler']
    threshold_path = config['paths']['threshold']
    plot_path = config['paths']['logs_recons_errors']
    
    # Create directories
    os.makedirs('src/core/models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Preprocess data
    train_data, scaler = preprocess_data(train_data_path, window_size=window_size)
    val_data, scaler = preprocess_data(val_data_path, window_size=window_size)
    train_data = torch.FloatTensor(train_data).to(device)
    val_data = torch.FloatTensor(val_data).to(device)
    
    # Initialize model
    model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    
    # Train model
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    model = train_model(model, train_data, val_data, epochs, batch_size)
    
    # Save model and scaler
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Compute and save threshold
    threshold = compute_threshold(model, val_data)
    with open(threshold_path, 'w') as f:
        json.dump({'threshold': threshold}, f)
    
    # Generate Plotly visualization
    plot_reconstruction_errors(val_data, model, threshold, plot_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Threshold saved to {threshold_path}")
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()