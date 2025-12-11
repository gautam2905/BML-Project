import torch
from torch.func import vmap, jacrev
from .bll import MarginalizeBLL

class LDBLL(MarginalizeBLL):
    """
    Latent Derivative BLL.
    Inherits the 'Marginalize' structure but adds the Derivative KL penalty.
    """
    def compute_derivative_reg(self, x, beta=0.1, gamma=0.1):
        """
        Computes the Forward KL divergence: KL( Prior || Posterior ).
        Normalized by sigma_n^2 to ensure scale invariance.
        """
        noise = torch.randn_like(x) * gamma
        s = x + noise
        
        def get_phi(x_in):
            return self.net(x_in.unsqueeze(0)).squeeze(0)
            
        J_phi = vmap(jacrev(get_phi))(s)
        if J_phi.ndim == 2: J_phi = J_phi.unsqueeze(2)
        
        # z_mean shape: [Batch, Input_Dim]
        z_mean = torch.einsum("n f d, f k -> n d", J_phi, self.post_mean) 
        # z_var shape: [Batch, Input_Dim] (Diagonal variance only)
        z_var = torch.einsum("n f d, f h, n h d -> n d", J_phi, self.post_cov, J_phi)
        
        prior_var = torch.ones_like(z_var) 
        
        z_var_clamped = torch.clamp(z_var, min=1e-6)
        
        # 5. Forward KL: KL( p(z) || q(z|x) ) = KL( Prior || Posterior )
        # Gaussians: 0.5 * ( (var_p / var_q) + (mu_q - mu_p)^2 / var_q - 1 + log(var_q / var_p) )
        
        # Ratio of variances
        var_ratio = prior_var / z_var_clamped
        
        # Squared difference of means (mu_p is 0)
        mean_diff_sq = z_mean.pow(2)
        
        # KL Term
        kl = 0.5 * ( var_ratio + (mean_diff_sq / z_var_clamped) - 1 + torch.log(z_var_clamped / prior_var) )
        
        return beta * kl.mean()