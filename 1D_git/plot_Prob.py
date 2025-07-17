
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Parameters
L = 6          # Domain [0, L]
delta_t = 0.1  # Time step

# NN Setup
x_dim = 1
sde_dt = 0.1
figdir = os.path.join(f'fig_{int(sde_dt*100):02d}/')
root_data = os.path.join(f'data_{int(sde_dt * 100):02d}/')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def exit_probability_eq31(T, x, L, num_terms=5000):
    """
    Exit probability by time T starting from position x
    Using equation (31): P(T,x) = 1 - sum_{n=1}^∞ (4/nπ) sin(nπx/L) exp(-1/2 (nπ/L)² T)
    """
    if x <= 0 or x >= L:
        return 1.0  # Already at boundary
    
    sum_terms = 0.0
    for n in range(1, num_terms + 1):
        if n % 2 != 0:
            coeff = 4 / (n * np.pi)
            sin_term = np.sin(n * np.pi * x / L)
            exp_term = np.exp(-0.5 * (n * np.pi / L)**2 * T)
            sum_terms += coeff * sin_term * exp_term
    
    return 1 - sum_terms

def load_weight_checkpoint(filepath):
    """Load checkpoint"""
    checkpoint = torch.load(filepath, weights_only=True)
    model = EscapeModel().to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_weight_model():
    """Load model"""
    savedir = root_data
    filename = 'NN_time_weight'
    
    print(savedir + filename + '.pt')
    model = load_weight_checkpoint(savedir + filename + '.pt')
    model.to(device)
    model.eval()
    return model

# Define NN model
class EscapeModel(nn.Module):
    def __init__(self):
        super(EscapeModel, self).__init__()
        self.dim = 1
        self.hid_size = 256
        self.net = nn.Sequential(
            nn.Linear(self.dim, self.hid_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(self.hid_size, self.hid_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(self.hid_size, self.hid_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(self.hid_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Calculate True Solution
x_positions_true = np.linspace(0.1, L-0.1, 101)
exit_probs_true = [exit_probability_eq31(delta_t, x, L) for x in x_positions_true]

print(x_positions_true)
print(exit_probs_true)

# Add boundary points for true solution
x_positions_true_full = np.concatenate([[0], x_positions_true, [L]])
exit_probs_true_full = np.concatenate([[1], exit_probs_true, [1]])

# Load NN model and calculate NN approximation
try:
    # Load normalization parameters
    with open(root_data + 'weight_mean_std.npy', 'rb') as f:
        mu_weight = np.load(f)
        s_weight = np.load(f)
    
    # Load trained model
    Escape = load_weight_model()
    
    # Generate NN predictions
    x_min, x_max = 0.1, 6-0.1
    num_points = 101
    P = np.linspace(x_min, x_max, num_points)
    
    # Prepare input for NN
    true_init_with_time = np.stack([P.ravel()], axis=1)
    test0 = (true_init_with_time - mu_weight) / s_weight
    test0 = torch.tensor(test0, dtype=torch.float32).to(device)
    
    # Get NN predictions
    with torch.no_grad():
        probabilities = Escape(test0).to('cpu').detach().numpy().reshape(P.shape)
    
    nn_exit_probs = 1 - probabilities
    print(nn_exit_probs)
    # Add boundary points for NN solution
    P_full = np.concatenate([[0], P, [L]])
    nn_exit_probs_full = np.concatenate([[1], nn_exit_probs, [1]])
    
    nn_available = True
    
except Exception as e:
    print(f"Could not load NN model: {e}")
    print("Plotting only the true solution...")
    nn_available = False

# Create combined plot
plt.figure(figsize=(12, 8))

# Plot true solution
plt.plot(x_positions_true_full, exit_probs_true_full, 'b-', linewidth=2, 
         label=f'True Solution (Eq. 31) with T = {delta_t}')

# Plot NN approximation if available
if nn_available:
    plt.plot(P_full, nn_exit_probs_full, 'r--', linewidth=2, 
             label='Neural Network Approximation')
    
    # Calculate and display error metrics (excluding boundaries)
    # Interpolate to same grid for comparison
    from scipy.interpolate import interp1d
    
    # Use interior points for error calculation
    x_common = x_positions_true  # Use true solution grid
    true_interp = np.array([exit_probability_eq31(delta_t, x, L) for x in x_common])
    
    # Interpolate NN solution to same grid
    nn_interp_func = interp1d(P, nn_exit_probs, kind='linear', fill_value='extrapolate')
    nn_interp = nn_interp_func(x_common)
    
    # Calculate error metrics
    mse = np.mean((true_interp - nn_interp)**2)
    mae = np.mean(np.abs(true_interp - nn_interp))
    max_error = np.max(np.abs(true_interp - nn_interp))
    
    print(f"Error Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Max Error: {max_error:.6f}")
    
    # Add error info to plot
    plt.text(0.02, 0.98, f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nMax Error: {max_error:.6f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.xlabel('Starting Position x', fontsize=16)
plt.ylabel('Exit Probability P(T,x)', fontsize=16)
plt.title(f'Exit Probability Comparison: True Solution vs Neural Network\nTime T = {delta_t}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(0, L)
plt.ylim(0, 1.05)

# Add some styling
plt.tick_params(labelsize=12)
plt.tight_layout()

# Save the plot
os.makedirs(figdir, exist_ok=True)
plt.savefig(os.path.join(figdir, 'combined_probability_comparison.png'), 
            dpi=300, bbox_inches='tight')
plt.savefig('combined_probability_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print some statistics
print(f"\nTrue Solution Statistics:")
print(f"Min exit probability: {np.min(exit_probs_true_full):.6f}")
print(f"Max exit probability: {np.max(exit_probs_true_full):.6f}")
print(f"Exit probability at center (x=3): {exit_probability_eq31(delta_t, 3, L):.6f}")

if nn_available:
    print(f"\nNN Approximation Statistics:")
    print(f"Min exit probability: {np.min(nn_exit_probs_full):.6f}")
    print(f"Max exit probability: {np.max(nn_exit_probs_full):.6f}")
    center_idx = len(P) // 2
    print(f"Exit probability at center (x≈3): {nn_exit_probs[center_idx]:.6f}")