import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset import SinGapDataset, get_test_grid, DEVICE
from src.bll import MarginalizeBLL
from src.ldbll import LDBLL
from src.vbll import VariationalBLL
from src.hybrid import HybridLDBLL

def train_marginalize_ldbll(model, dataset, epochs=2000, lr=0.001, is_ldbll=False, beta=0.01):
    """
    Unified training loop for MarginalizeBLL and LDBLL.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X, Y = dataset.X, dataset.Y
    losses = []  # <--- CHANGE 1: Init list
    
    pbar = tqdm(range(epochs), desc=f"Training {model.__class__.__name__}")
    
    for _ in pbar:
        opt.zero_grad()

        with torch.no_grad():
            model.update_posterior_stats(X, Y)
        
        # Minimize Expected NLL
        # 1. Forward Pass
        pred_mean, epistemic_var = model(X)
        sigma_n = torch.exp(model.log_sigma_noise)
        variance_noise = sigma_n**2

        # 2. Loss Calculation (Expected NLL)
        mse_term = (Y - pred_mean)**2
        var_term = epistemic_var 

        nll_loss = ((mse_term + var_term) / (2 * variance_noise)).mean() + 0.5 * torch.log(variance_noise)
        loss = nll_loss
        
        if is_ldbll:
            reg_loss = model.compute_derivative_reg(X, beta=beta, gamma=0.1)
            loss += reg_loss
            
            pbar.set_postfix({"NLL": nll_loss.item(), "Reg": reg_loss.item()})
        else:
             pbar.set_postfix({"NLL": nll_loss.item()})

        losses.append(loss.item()) # <--- CHANGE 2: Record loss
        loss.backward()
        opt.step()
        
    model.update_posterior_stats(X, Y)
    return model, losses

def train_variational(model, dataset, epochs=1000, lr=1e-3):
    # FIX 2: Correct Optimizer Groups to avoid double regularization
    # Feature extractor gets weight decay
    # Variational parameters (last layer) get NO weight decay (handled by KL/Prior)
    
    variational_params = {
        'w_mean', 'l_offdiag', 'l_log_diag', 'log_prec'
    }
    
    params_net = []
    losses = []
    params_var = []
    
    for name, param in model.named_parameters():
        if any(v_name in name for v_name in variational_params):
            params_var.append(param)
        else:
            params_net.append(param)

    # Use AdamW as per Appendix D.4 [cite: 938]
    opt = torch.optim.AdamW([
        {'params': params_net, 'weight_decay': 1e-2}, # Standard weight decay for features
        {'params': params_var, 'weight_decay': 0.0}   # Zero weight decay for VBLL params
    ], lr=lr)

    # Use the full batch if dataset is small (common in these toy gap examples)
    # or ensure KL weight is scaled by 1/N_total if using mini-batches.
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    N = len(dataset)
    
    model.train()
    pbar = tqdm(range(epochs), desc="Training VBLL")
    
    for _ in pbar:
        epoch_loss = 0
        loss_epoch = []
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Scale KL by 1/N (Total dataset size) [cite: 152]
            loss = model.compute_train_loss(x, y, kl_weight=1.0/N)
            # losses.append(loss.item()) # Record loss
            loss_epoch.append(loss.item())
            opt.zero_grad()
            loss.backward()
            
            # Gradient Clipping (Appendix B.4 [cite: 623])
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()
            epoch_loss += loss.item()
            
        loss_episode = np.array(loss_epoch).mean()
        losses.append(loss_episode)
        pbar.set_postfix({'loss': epoch_loss / len(loader)})
            
    return model, losses

def train_hybrid(model, dataset, epochs=2000, lr=0.005, beta=1.0):
    """
    Training loop for Hybrid VBLL + LDBLL.
    """
    # Use clip_grad_norm to prevent exploding gradients typical in derivative training
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Simple full-batch or mini-batch loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    N = len(dataset)
    
    pbar = tqdm(range(epochs), desc="Training Hybrid Model")
    
    for _ in pbar:
        total_nll = 0
        total_reg = 0
        
        for x, y in loader:
            # 1. VBLL Objective: Expected Log Likelihood + Parameter KL
            # This trains the model to fit data and maintain weight uncertainty
            # We scale KL by 1/N because compute_train_loss returns sum-like scale for NLL
            vbll_loss = model.compute_train_loss(x, y, kl_weight=1.0/N)
            
            # 2. LDBLL Regularization: Derivative KL
            # This forces the gradient of the function to be "smooth" (close to 0 prior) 
            # where data is sparse, preventing overfitting.
            ld_reg = model.compute_derivative_reg(x, beta=beta, gamma=0.1)
            
            loss = vbll_loss + ld_reg
            
            opt.zero_grad()
            loss.backward()
            
            # Critical for stability when optimizing derivatives
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            opt.step()
            
            total_nll += vbll_loss.item()
            total_reg += ld_reg.item()
            
        pbar.set_postfix({"VBLL_Loss": total_nll / len(loader), "LD_Reg": total_reg / len(loader)})
            
    return model

def visualize_models(models_dict, dataset, device, save_name="model_comparison.png"):
    """
    Unified plotter for any number of BLL models.
    
    Args:
        models_dict: Dictionary { "Title": model_instance }
        dataset: The training dataset
        device: torch device
        save_name: Filename to save the plot
    """
    # 1. Setup Test Grid (Extrapolating beyond training data)
    # Range is -6 to 6 to show OOD behavior (assuming data is approx -4 to 4)
    x_test = torch.linspace(-6, 6, 400).unsqueeze(1).to(device)
    x_np = x_test.cpu().numpy().flatten()
    
    # 2. Setup Plot Layout
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True, dpi=100)
    if n_models == 1: axes = [axes] # Handle single model case

    # 3. Iterate and Plot
    for i, (name, model) in enumerate(models_dict.items()):
        ax = axes[i]
        model.eval()
        
        with torch.no_grad():
            # Get model predictions
            mean, var = model(x_test)
            
            # Convert to numpy for plotting
            mean = mean.cpu().numpy().flatten()
            std = torch.sqrt(var).cpu().numpy().flatten()
            
            # Extract Training Data
            train_x = dataset.X.cpu().numpy()
            train_y = dataset.Y.cpu().numpy()

        # A. Highlight the "Gap" (No Data Region)
        # Assuming SinGap typically has a gap between -1 and 1 or -2 and 2. 
        # We define it visually here for context.
        ax.axvspan(-2, 2, color='mistyrose', alpha=0.4, label='Data Gap (OOD)')

        # B. Plot Training Data
        ax.scatter(train_x, train_y, c='k', s=15, alpha=0.6, label='Train Data', zorder=3)

        # C. Plot Predictive Mean
        ax.plot(x_np, mean, color='royalblue', linewidth=2, label='Pred Mean', zorder=2)

        # D. Plot Uncertainty (2 Sigma / 95% Confidence)
        ax.fill_between(x_np, mean - 2*std, mean + 2*std, 
                        color='royalblue', alpha=0.25, label='Uncertainty ($2\sigma$)', zorder=1)

        # Styling
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_ylim(-5, 5) # Fix Y-axis to keep comparisons valid
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Only add legend to the first plot to avoid clutter
        if i == 0:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    print(f"Saving visualization to {save_name}...")
    plt.savefig(save_name)
    plt.show()

def plot_loss_curves(losses_dict, save_name="loss_curves.png"):
    """
    Plots training loss curves for multiple models.
    losses_dict: {'Model Name': [loss_values_list]}
    """
    plt.figure(figsize=(10, 5))
    
    for name, loss_hist in losses_dict.items():
        # Plot only the last 90% to avoid initial huge spikes scaling the graph out
        # or plot everything if short
        start_idx = 0 if len(loss_hist) < 100 else 10
        plt.plot(range(start_idx, len(loss_hist)), loss_hist[start_idx:], label=name, linewidth=2)

    plt.title("Training Loss Convergence")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    print(f"Saving loss curves to {save_name}...")
    plt.savefig(save_name)
    plt.show()

if __name__ == "__main__":
    # Dummy Dataset for testing
    ds = SinGapDataset()
    
    # 1. Train Marginalize BLL
    
    # print("\n--- Training Marginalize BLL ---")
    # model_bll = MarginalizeBLL().to(DEVICE)
    # model_bll, bll_losses = train_marginalize_ldbll(model_bll, ds, is_ldbll=False)
    
    # # 2. Train LDBLL
    # print("\n--- Training LDBLL ---")
    # model_ldbll = LDBLL().to(DEVICE)
    # # Increased beta slightly for better visual effect of regularization
    # model_ldbll, ldbll_losses = train_marginalize_ldbll(model_ldbll, ds, is_ldbll=True, beta=0.005, epochs=4500)
    
    # model_ldbll_less = LDBLL().to(DEVICE)
    # # Increased beta slightly for better visual effect of regularization
    # model_ldbll_less, ldbll_losses_less = train_marginalize_ldbll(model_ldbll_less, ds, is_ldbll=True, beta=0.005, epochs=2000)

    # # 3. Visualize Comparison
    # # You can simply pass the models in a dictionary
    # models_to_plot = {
    #     'Marginalize BLL (Baseline)': model_bll,
    #     'LDBLL (Derivative Reg)': model_ldbll,
    #     'LDBLL (Less Training)': model_ldbll_less
    # }
    # visualize_models(models_to_plot, ds, DEVICE, save_name="plots/bll_vs_ldbll_compare.png")

    # losses_to_plot = {
    #     'Marginalize BLL': bll_losses,
    #     'LDBLL': ldbll_losses,
    #     'LDBLL (Less Training)': ldbll_losses_less
    # }
    # plot_loss_curves(losses_to_plot, save_name="plots/bll_vs_ldbll_loss_curves.png")


    ds = SinGapDataset(num_samples=2000, gap_bounds=(-4, 4))

    print("\n--- Training Hybrid ---")
    vbll = VariationalBLL().to(DEVICE)
    vbll, vbll_losses = train_variational(vbll, ds, epochs=2000) 
    
    # hybrid = HybridLDBLL().to(DEVICE)
    # hybrid = train_hybrid(hybrid, ds, beta=0.005, epochs=2000)
    
    visualize_models({
       'Variational BLL': vbll, 
    #    'Hybrid LDBLL': hybrid
    }, ds, DEVICE, save_name="plots/hybrid_compare.png")

    plot_loss_curves({
        'Variational BLL': vbll_losses,
    #    'Hybrid LDBLL': hybrid_losses
    }, save_name="plots/hybrid_loss_curves.png")
