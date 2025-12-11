import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.utils.spectral_norm as sn 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VariationalBLL(nn.Module):
    def __init__(self, in_dim=1, feature_dim=50, hidden_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 1. Feature Extractor
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, feature_dim) 
        )

        # self.net = nn.Sequential(
        #     sn(nn.Linear(in_dim, hidden_dim)),
        #     nn.LeakyReLU(), 
        #     sn(nn.Linear(hidden_dim, hidden_dim)),
        #     nn.LeakyReLU(),
        #     sn(nn.Linear(hidden_dim, feature_dim)) 
        # )

        
        self.w_mean = nn.Parameter(torch.zeros(self.feature_dim, 1))
        
        self.l_offdiag = nn.Parameter(torch.zeros(self.feature_dim, self.feature_dim))
        self.l_log_diag = nn.Parameter(torch.zeros(self.feature_dim)) 
        
        self.log_prec = nn.Parameter(torch.tensor(0.0)) 

        self.prior_nu = 1.0 
        self.prior_scale = 1.0
        
        self.to(DEVICE) # Ensure model moves to device

    def get_features(self, x):
        """Computes phi(x) and appends a bias column of 1s"""
        phi = self.net(x)
        return phi
    
    def get_post_cov(self):
        mask = torch.tril(torch.ones_like(self.l_offdiag), diagonal=-1)
        L = self.l_offdiag * mask
        L = L + torch.diag(torch.exp(self.l_log_diag))
        S = L @ L.T
        return S, L

    def forward(self, x):
        phi = self.get_features(x)
        S, _ = self.get_post_cov()
        
        # Correct variable name (log_prec)
        precision = torch.exp(self.log_prec)
        sigma_sq = 1.0 / precision
        
        # Predictive Mean: y = phi @ w
        pred_mean = phi @ self.w_mean
        
        # Epistemic Variance: phi @ S @ phi.T (taking diag)
        # Efficient computation: sum((phi @ S) * phi)
        epistemic = torch.sum((phi @ S) * phi, dim=1, keepdim=True)
        
        # Return Mean and Total Variance (Epistemic + Aleatoric)
        return pred_mean, epistemic + sigma_sq

    def compute_train_loss(self, x, y, kl_weight=1.0):
        phi = self.get_features(x)
        S, L = self.get_post_cov()
        
        precision = torch.exp(self.log_prec)

        # --- 1. Expected Log Likelihood (Theorem 1, Eq 13)  ---
        # Note: We minimize Negative ELBO. 
        # Loss terms should be: - (LogLikelihood) + TraceTerm
        
        pred_mean = phi @ self.w_mean
        mse = (y - pred_mean)**2
        
        # Negative Log Gaussian (ignoring constant terms for optimization if desired, 
        # but kept here for correctness)
        # NLL = 0.5 * log(2pi) - 0.5 * log(precision) + 0.5 * mse * precision
        nll_mean = 0.5 * np.log(2 * np.pi) - 0.5 * self.log_prec + 0.5 * mse * precision
        
        # Trace term: 0.5 * tr(phi^T S phi Sigma^-1)
        # = 0.5 * precision * diag(phi S phi^T)
        epistemic_var = torch.sum((phi @ S) * phi, dim=1, keepdim=True)
        trace_term = 0.5 * epistemic_var * precision
        
        # Averaged over batch (standard for SGD)
        expected_nll = (nll_mean + trace_term).mean()

        # --- 2. KL Divergence q(w)||p(w) (Eq 46)  ---
        # Assuming Prior W ~ N(0, I/s) where s=1 (isotropic standard normal)
        tr_S = torch.sum(L ** 2) 
        mu_sq = torch.sum(self.w_mean**2)
        k = self.feature_dim # Adjusted for bias
        log_det_S = 2 * torch.sum(self.l_log_diag)
        
        kl_last_layer = 0.5 * (tr_S + mu_sq - k - log_det_S)

        # --- 3. Regularization on Sigma (Eq 51)  ---
        # Minimizing negative log prior of Inverse Wishart
        nu_tilde = self.prior_nu + 1 + 1 # nu + N + 1 (N=1 for scalar)
        reg_sigma = 0.5 * (self.prior_scale * precision - nu_tilde * self.log_prec)
        
        # Total Loss
        return expected_nll + kl_weight * (kl_last_layer + reg_sigma)