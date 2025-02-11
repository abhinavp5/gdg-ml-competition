import logging
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Define improved model using Sequential API
class MyModel(nn.Module):
    def __init__(self, input_dim=397, hidden1=512, hidden2=256, hidden3=128, hidden4 =64, dropout_prob=0.3):
        super(MyModel, self).__init__()
        
        self.model = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(p=dropout_prob),
            
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(p=dropout_prob),
            
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden3),
            nn.Dropout(p=dropout_prob),

            nn.Linear(hidden3, hidden4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden4),
            nn.Dropout(p=dropout_prob),
            
            nn.Linear(hidden4, 1)
        )
        
    def forward(self, x):
        return self.model(x)

def create_model(features):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()

    # Ensure features are moved to CPU before applying StandardScaler
    features_cpu = features.cpu().numpy()
    features_scaled = scaler.fit_transform(features_cpu)

    # Convert back to tensor and move to CUDA if available
    features_scaled = torch.tensor(features_scaled, dtype=torch.float32, device=device)

    # Move model to CUDA
    model = MyModel(features_scaled.shape[1]).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

    return model, optimizer


if __name__ == '__main__':
    # create sample model with 228 input features
    model, _ = create_model(torch.zeros(1, 228))
    print(model)
