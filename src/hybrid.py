import torch
import torch.nn as nn
from torch.func import vmap, jacrev
from .vbll import VariationalBLL  # Assumes the CORRECTED VariationalBLL from previous step

class HybridLDBLL(VariationalBLL):
    """
    Hybrid Model:
    1. Uses Variational Inference for weights (VBLL architecture).
    2. Adds the Derivative Regularization term (LDBLL prior) for OOD robustness.
    """
    def compute_derivative_reg(self, x, beta=1.0, gamma=0.1):
        """
        Computes LDBLL Regularization using FORWARD KL.
        Minimizing KL( Prior || Posterior ) where Prior ~ N(0, I).
        
        Effect:
        1. Penalizes large slopes (mu_grad) -> Smoothes the mean (removes sharp jump).
        2. Penalizes small variance (var_grad) -> Inflates uncertainty in the gap.
        """
        # 1. Perturb input to "touch" the gap
        # If the gap is wide (e.g., -2 to 2), gamma must be large enough 
        # for these samples to fall INTO the gap. 
        # Try gamma=0.2 or 0.3 if the jump persists.
        noise = torch.randn_like(x) * gamma 
        s = x + noise
        
        # 2. Compute Jacobian J_phi (same as before)
        def get_phi(x_in):
            return self.net(x_in.unsqueeze(0)).squeeze(0)
            
        J_phi = vmap(jacrev(get_phi))(s)
        if J_phi.ndim == 4: J_phi = J_phi.squeeze(1)
            
        # 3. Compute Posterior of the Gradient (z) (same as before)
        J_T = J_phi.transpose(1, 2)
        S, _ = self.get_post_cov()
        
        # Mean Slope (The cause of the sharp jump)
        mu_grad = J_T @ self.w_mean 
        
        # Variance of Slope (The cause of the pinched uncertainty)
        # sum((J^T @ S) * J^T, dim=2)
        var_grad = torch.sum((J_T @ S) * J_T, dim=2, keepdim=True)
        
        # 4. FORWARD KL Implementation
        # Formula: 0.5 * ( (var_prior + (mu_post - mu_prior)^2) / var_post  + log(var_post) - 1 )
        # With Prior N(0, 1):
        # term1 = (1 + mu_grad^2) / var_grad
        
        eps = 1e-6
        
        # This term punishes the model if the variance is small.
        # It ALSO punishes the model if mu_grad is large (steep slope).
        trace_term = (1.0 + mu_grad**2) / (var_grad + eps)
        
        log_term = torch.log(var_grad + eps)
        
        # Total KL
        kl = 0.5 * (trace_term + log_term - 1.0)
        
        return beta * kl.mean()