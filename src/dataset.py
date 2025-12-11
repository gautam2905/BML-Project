import torch
import numpy as np
from torch.utils.data import Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SinGapDataset(Dataset):
    def __init__(self, num_samples=100, noise=0.1, gap_bounds=(-2, 2), seed=42, manual_gap_points=None):
        self.num_samples = num_samples
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 1. Generate split data
        n_half = num_samples // 2
        x_left = np.random.uniform(-6, gap_bounds[0], (n_half, 1))
        x_right = np.random.uniform(gap_bounds[1], 6, (n_half, 1))
        
        self.X = np.vstack([x_left, x_right])
        
        # 2. Add manual points if provided
        if manual_gap_points is not None:
            x_manual = np.array(manual_gap_points).reshape(-1, 1)
            self.X = np.vstack([self.X, x_manual])

        # 3. Generate Y for all points (main + manual)
        self.Y = np.sin(self.X) + noise * np.random.randn(len(self.X), 1)
        
        # Move to GPU
        self.X = torch.tensor(self.X, dtype=torch.float32).to(DEVICE)
        self.Y = torch.tensor(self.Y, dtype=torch.float32).to(DEVICE)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class VariableDensitySinDataset(Dataset):
    def __init__(self, num_samples=150, noise=0.1, bounds=(-6, 6), sparse_bounds=(-2, 2), density_ratio=0.1, seed=42):
        """
        Args:
            num_samples: Total number of samples.
            noise: Standard deviation of additive Gaussian noise.
            bounds: The full range of the dataset (min, max).
            sparse_bounds: The middle region (min, max) where data should be sparse.
            density_ratio: Ratio of points in the sparse region vs the dense regions. 
                           e.g., 0.1 means the sparse region has 10% the density of outer regions.
        """
        self.num_samples = num_samples
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Calculate lengths of regions
        total_len = bounds[1] - bounds[0]
        sparse_len = sparse_bounds[1] - sparse_bounds[0]
        dense_len = total_len - sparse_len
        
        # Calculate number of samples for each region based on density target
        # We solve for x (samples per unit length in dense region):
        # x * dense_len + (x * density_ratio) * sparse_len = num_samples
        dense_density = num_samples / (dense_len + density_ratio * sparse_len)
        sparse_density = dense_density * density_ratio
        
        n_sparse = int(sparse_density * sparse_len)
        n_dense = num_samples - n_sparse # Remainder goes to dense to ensure total matches num_samples
        n_dense_left = n_dense // 2
        n_dense_right = n_dense - n_dense_left # Handle odd numbers
        
        # Generate Data
        # Dense Left: [-6, -2]
        x_dense_1 = np.random.uniform(bounds[0], sparse_bounds[0], (n_dense_left, 1))
        
        # Sparse Middle: [-2, 2]
        x_sparse = np.random.uniform(sparse_bounds[0], sparse_bounds[1], (n_sparse, 1))
        
        # Dense Right: [2, 6]
        x_dense_2 = np.random.uniform(sparse_bounds[1], bounds[1], (n_dense_right, 1))
        
        # Combine and Sort (sorting helps visualization, usually strictly not needed for training but good for sanity)
        self.X = np.vstack([x_dense_1, x_sparse, x_dense_2])
        
        # Shuffle implies "not hard separated" order-wise, though spatially density varies
        np.random.shuffle(self.X)
        
        self.Y = np.sin(self.X) + noise * np.random.randn(num_samples, 1)
        
        # Move to GPU immediately
        self.X = torch.tensor(self.X, dtype=torch.float32).to(DEVICE)
        self.Y = torch.tensor(self.Y, dtype=torch.float32).to(DEVICE)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class MultiGapSinDataset(Dataset):
    def __init__(self, num_samples=300, noise=0.1, bounds=(-6, 6), 
                 sparse_regions=[(-5, -3), (-1, 1), (3, 5)], 
                 density_ratio=0.05, seed=42):
        """
        Args:
            sparse_regions: List of tuples [(start, end), ...] defining areas with less data.
            density_ratio: Ratio of data points in sparse regions vs dense regions.
                           0.05 means sparse regions have 5% density of normal regions.
        """
        self.num_samples = num_samples
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 1. Determine Dense Regions by subtracting sparse regions from bounds
        # Assumes sparse_regions are sorted and non-overlapping
        dense_regions = []
        current_start = bounds[0]
        sparse_regions = sorted(sparse_regions, key=lambda x: x[0])
        
        for (s_start, s_end) in sparse_regions:
            if s_start > current_start:
                dense_regions.append((current_start, s_start))
            current_start = max(current_start, s_end)
            
        if current_start < bounds[1]:
            dense_regions.append((current_start, bounds[1]))
            
        # 2. Calculate Total Lengths to distribute points
        total_sparse_len = sum(e - s for s, e in sparse_regions)
        total_dense_len = sum(e - s for s, e in dense_regions)
        
        # Solve for x (density): x * dense_len + (x * ratio) * sparse_len = num_samples
        dense_density = num_samples / (total_dense_len + density_ratio * total_sparse_len)
        sparse_density = dense_density * density_ratio
        
        # 3. Generate Data
        X_parts = []
        
        # Generate Dense Points
        for (start, end) in dense_regions:
            n = int(dense_density * (end - start))
            if n > 0:
                X_parts.append(np.random.uniform(start, end, (n, 1)))
            
        # Generate Sparse Points
        for (start, end) in sparse_regions:
            n = int(sparse_density * (end - start))
            # Ensure a tiny bit of data exists if density is not 0 (probabilistic)
            if n == 0 and density_ratio > 0 and np.random.rand() < 0.5:
                n = 1
            if n > 0:
                X_parts.append(np.random.uniform(start, end, (n, 1)))
                
        self.X = np.vstack(X_parts)
        
        # Fix rounding errors to match num_samples exactly
        if len(self.X) < num_samples:
            # Fill remaining with random dense points
            needed = num_samples - len(self.X)
            dr = dense_regions[0] # Pick first dense region
            self.X = np.vstack([self.X, np.random.uniform(dr[0], dr[1], (needed, 1))])
        elif len(self.X) > num_samples:
            # Randomly trim
            indices = np.random.choice(len(self.X), num_samples, replace=False)
            self.X = self.X[indices]
            
        # Shuffle
        np.random.shuffle(self.X)
        
        self.Y = np.sin(self.X) + noise * np.random.randn(len(self.X), 1)
        
        self.X = torch.tensor(self.X, dtype=torch.float32).to(DEVICE)
        self.Y = torch.tensor(self.Y, dtype=torch.float32).to(DEVICE)
        
        # Store regions for plotting later
        self.sparse_regions = sparse_regions

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Helper to visualize the density difference
if __name__ == "__main__":
    ds = VariableDensitySinDataset(num_samples=200, density_ratio=0.1)
    print(f"Dataset created with {len(ds)} samples.")
    print(f"Shape of X: {ds.X.shape}")
    
    # Simple ASCII hist
    import matplotlib.pyplot as plt
    X_cpu = ds.X.cpu().numpy().flatten()
    plt.figure(figsize=(10, 3))
    plt.hist(X_cpu, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Data Density Distribution")
    plt.xlabel("X value")
    plt.ylabel("Count")
    plt.show()