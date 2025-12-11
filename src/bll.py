import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Assuming DEVICE is defined elsewhere, e.g.:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MarginalizeBLL(nn.Module):
    def __init__(self, in_dim=1, feature_dim=50, hidden_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.log_sigma_noise = nn.Parameter(torch.tensor(np.log(0.1))) 
        self.log_lambda_prior = nn.Parameter(torch.tensor(0.0), requires_grad=False) 

        self.register_buffer("post_mean", torch.zeros(feature_dim, 1)) # mu_n
        self.register_buffer("post_cov", torch.eye(feature_dim)) # Λ_n

    def get_features(self, x):
        return self.net(x)

    @torch.no_grad()
    def update_posterior_stats(self, X, y):
        self.eval() 
        Phi = self.get_features(X)
        sigma_n = torch.exp(self.log_sigma_noise)
        lambda_0 = torch.exp(self.log_lambda_prior)
        
        eye = torch.eye(self.feature_dim, device=X.device)
        A = (1/sigma_n**2) * (Phi.T @ Phi) + lambda_0 * eye
        
        try:
            L = torch.linalg.cholesky(A)
            self.post_cov = torch.cholesky_inverse(L)
        except RuntimeError:
            self.post_cov = torch.linalg.inv(A)
            
        self.post_mean = (1/sigma_n**2) * self.post_cov @ (Phi.T @ y)
        self.train() 

    def forward(self, x):
        phi = self.get_features(x)
        pred_mean = phi @ self.post_mean
        epistemic_var = torch.sum((phi @ self.post_cov) * phi, dim=1, keepdim=True)
        return pred_mean, epistemic_var