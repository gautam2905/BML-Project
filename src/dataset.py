import torch
import numpy as np
from torch.utils.data import Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SinGapDataset(Dataset):
    def __init__(self, num_samples=100, noise=0.1, gap_bounds=(-2, 2), seed=42):
        self.num_samples = num_samples
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate split data to create a clean gap
        n_half = num_samples // 2
        x_left = np.random.uniform(-6, gap_bounds[0], (n_half, 1))
        x_right = np.random.uniform(gap_bounds[1], 6, (n_half, 1))
        
        self.X = np.vstack([x_left, x_right])
        self.Y = np.sin(self.X) + noise * np.random.randn(num_samples, 1)
        
        # Move to GPU immediately for full-batch methods
        self.X = torch.tensor(self.X, dtype=torch.float32).to(DEVICE)
        self.Y = torch.tensor(self.Y, dtype=torch.float32).to(DEVICE)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_test_grid(x_min=-7, x_max=7, n_points=400):
    x = torch.linspace(x_min, x_max, n_points).unsqueeze(1).to(DEVICE)
    y = torch.sin(x)
    return x, y