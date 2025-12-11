import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset import SinGapDataset, DEVICE, VariableDensitySinDataset, MultiGapSinDataset
from src.bll import MarginalizeBLL
from src.ldbll import LDBLL
from src.vbll import VariationalBLL
from src.hybrid import VariationalLDBLL

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
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
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

def train_variational_ldbll(model, dataset, epochs=1000, lr=1e-3, ld_beta=0.1):
    # Optimizer setup (Same as before)
    variational_params = {'w_mean', 'l_offdiag', 'l_log_diag', 'log_prec'}
    params_net = []
    params_var = []
    losses = []
    
    for name, param in model.named_parameters():
        if any(v_name in name for v_name in variational_params):
            params_var.append(param)
        else:
            params_net.append(param)

    opt = torch.optim.AdamW([
        {'params': params_net, 'weight_decay': 1e-2},
        {'params': params_var, 'weight_decay': 0.0} 
    ], lr=lr)

    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    N = len(dataset)
    
    model.train()

    pbar = tqdm(range(epochs), desc="Training Variational LDBLL")
    for _ in pbar:
        epoch_loss = 0
        loss_epoch = []
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # 1. Compute Standard VBLL Loss (ELBO)
            # kl_weight scaled by 1/N as per standard practice
            vbll_loss = model.compute_train_loss(x, y, kl_weight=1.0/N)
            
            # 2. Compute Latent Derivative Regularization
            # This uses the perturbed samples internally
            ld_loss = model.compute_derivative_reg(x, beta=ld_beta, gamma=0.1)
            
            # 3. Total Loss
            loss = vbll_loss + ld_loss
            loss_epoch.append(loss.item())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            epoch_loss += loss.item()

        losses.append(np.array(loss_epoch).mean())
        pbar.set_postfix({'loss': epoch_loss / len(loader)})

        # if epoch % 100 == 0:
            # print(f"Epoch {epoch} | Total Loss: {epoch_loss/len(loader):.4f} | VBLL: {vbll_loss.item():.4f} | LD: {ld_loss.item():.4f}")

    return model, losses

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

    # ds = SinGapDataset(num_samples=500)
    # # ds = VariableDensitySinDataset(num_samples=500, density_ratio=0.1)
    # # ds = MultiGapSinDataset(num_samples=500)
    
    # # 1. Train Marginalize BLL
    
    # print("\n--- Training Marginalize BLL ---")
    # model_bll = MarginalizeBLL().to(DEVICE)
    # model_bll, bll_losses = train_marginalize_ldbll(model_bll, ds, is_ldbll=False)
    
    # # 2. Train LDBLL
    # print("\n--- Training LDBLL ---")
    # model_ldbll = LDBLL().to(DEVICE)
    # model_ldbll, ldbll_losses = train_marginalize_ldbll(model_ldbll, ds, is_ldbll=True, beta=0.005, epochs=4500)
    
    # model_ldbll_less = LDBLL().to(DEVICE)
    # model_ldbll_less, ldbll_losses_less = train_marginalize_ldbll(model_ldbll_less, ds, is_ldbll=True, beta=0.005, epochs=2000)

    # models_to_plot = {
    #     'Marginalize BLL (Baseline)': model_bll,
    #     'LDBLL (Derivative Reg)': model_ldbll,
    #     'LDBLL (Less Training)': model_ldbll_less
    # }

    # losses_to_plot = {
    #     'Marginalize BLL': bll_losses,
    #     'LDBLL': ldbll_losses,
    #     'LDBLL (Less Training)': ldbll_losses_less
    # }


    ds = SinGapDataset(num_samples=500)
    # ds = VariableDensitySinDataset(num_samples=500, density_ratio=0.1)

    print("\n--- Training Hybrid ---")
    vbll = VariationalBLL().to(DEVICE)
    vbll, vbll_losses = train_variational(vbll, ds, epochs=4000) 

    print("\n--- Training Hybrid LDBLL ---")
    hybrid = VariationalLDBLL().to(DEVICE)
    hybrid, hybrid_losses = train_variational_ldbll(hybrid, ds, ld_beta=0.005, epochs=5000)
    
    # Plotting Results

    visualize_models(models_to_plot, ds, DEVICE, save_name="plots/bll_vs_ldbll_compare.png")
    
    plot_loss_curves(losses_to_plot, save_name="plots/bll_vs_ldbll_loss_curves.png")
    
    visualize_models({
       'Variational BLL': vbll, 
       'Hybrid Variational LDBLL': hybrid
    }, ds, DEVICE, save_name="plots/hybrid_compare.png")

    plot_loss_curves({
        'Variational BLL': vbll_losses,
       'Hybrid Variational LDBLL': hybrid_losses
    }, save_name="plots/hybrid_loss_curves.png")
