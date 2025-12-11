import torch
import torch.nn as nn
from torch.func import vmap, jacrev
from .vbll import VariationalBLL

class VariationalLDBLL(VariationalBLL): # Inherits from your fixed VBLL class
    def __init__(self, in_dim=1, feature_dim=50, hidden_dim=64):
        super().__init__(in_dim, feature_dim, hidden_dim)
        
    def compute_derivative_reg(self, x, beta=1.0, gamma=0.1):
        """
        Computes the Latent Derivative Regularization (Forward KL).
        
        Args:
            x: Input data batch
            beta: Weight of the regularization term
            gamma: Noise standard deviation for perturbation (Paper Sec 3 "Index Set")
        """
        noise = torch.randn_like(x) * gamma
        s = x + noise
        
        def get_phi(x_in):
            return self.net(x_in.unsqueeze(0)).squeeze(0)
            
        # J_phi shape: [Batch, Feature_Dim, Input_Dim]
        J_phi = vmap(jacrev(get_phi))(s)
        
        S, _ = self.get_post_cov() 
        
        # Mean derivative: J_phi^T @ w_mean
        # Shape: [Batch, Input_Dim]
        z_mean = torch.einsum("n f d, f k -> n d", J_phi, self.w_mean)
        
        # Variance of derivative: diag(J_phi^T @ S @ J_phi)
        # We only need the diagonal variance for the KL computation
        # Shape: [Batch, Input_Dim]
        z_var = torch.einsum("n f d, f h, n h d -> n d", J_phi, S, J_phi)       
        z_var_clamped = torch.clamp(z_var, min=1e-6)
        
        # 4. Define Prior over Derivatives: p(z)
        prior_var = torch.ones_like(z_var) 
        
        # 5. Compute Forward KL: KL( p(z) || q(z|s) )
        # Formula: 0.5 * ( (var_p / var_q) + (mu_q - mu_p)^2 / var_q - 1 + log(var_q / var_p) )
        
        var_ratio = prior_var / z_var_clamped
        mean_diff_sq = z_mean.pow(2)
        
        kl_div = 0.5 * ( 
            var_ratio + 
            (mean_diff_sq / z_var_clamped) - 
            1 + 
            torch.log(z_var_clamped / prior_var) 
        )
        
        return beta * kl_div.mean()