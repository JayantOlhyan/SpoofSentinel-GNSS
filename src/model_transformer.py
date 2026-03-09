import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os

WINDOW_SIZE = 10  # Look at 10 consecutive timesteps

class GNSSSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list, label_col: str = None, window: int = WINDOW_SIZE):
        """
        Groups data by PRN and creates sliding windows of size `window`.
        Each sample = (window x num_features) tensor.
        Label = label of the last timestep in the window.
        """
        self.sequences = []
        self.labels = []
        self.indices = [] # Keep track of original indices for alignment
        
        # Ensure data is sorted for temporal consistency
        df = df.sort_values(['PRN', 'RX_time']).reset_index()
        
        for prn, group in df.groupby('PRN'):
            if len(group) < window:
                continue
            
            features = group[feature_cols].values
            if label_col:
                labels = group[label_col].values
            
            # Sliding window creation
            for i in range(len(group) - window + 1):
                self.sequences.append(features[i:i+window])
                if label_col:
                    self.labels.append(labels[i+window-1])
                self.indices.append(group.iloc[i+window-1]['index'])
        
        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        if label_col:
            self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]

class GNSSSpoofTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, window, features)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # Take the last timestep's representation
        return self.classifier(x).squeeze(1)

def train_transformer(model, train_loader, val_loader=None, epochs=30, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Use BCE loss for binary classification
    criterion = nn.BCELoss()
    
    print(f"Training Temporal Transformer on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    os.makedirs('outputs', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/transformer_model.pt')
    print("Transformer model saved to outputs/transformer_model.pt")
    return model

def get_transformer_predictions(model, dataset: GNSSSequenceDataset) -> np.ndarray:
    """Return probability predictions as numpy array, aligned with dataset indices."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_preds = []
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, list):
                X_batch = batch[0]
            else:
                X_batch = batch
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds.extend(preds.cpu().numpy())
            
    return np.array(all_preds)
