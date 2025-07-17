import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
x_dim = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_output = 'data_2d/'   # where to save trained models
root_data = 'data_2d/'     # where the datasets are
figdir = 'figures/'        # where to save figures

def load_escape_checkpoint_2d(filepath):
    """
    Load checkpoint for 2D escape model
    """
    try:
        checkpoint = torch.load(filepath, weights_only=True, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        checkpoint = torch.load(filepath, map_location=device)
    
    model = EscapeModel2D().to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_normalization_params_2d(suffix=''):
    """
    Load normalization parameters for 2D model
    """
    filepath = os.path.join(root_data, f'weight_mean_std_2d{suffix}.npy')
    print(f"Loading normalization from: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            mu_weight = np.load(f)
            s_weight = np.load(f)
        return mu_weight, s_weight
    except FileNotFoundError:
        print(f"Warning: Normalization file {filepath} not found.")
        return None, None

def load_escape_model_2d(suffix=''):
    """
    Load the trained 2D escape model with optional suffix
    """
    savedir = root_data
    filename = f'NN_time_weight_2d{suffix}'
    
    filepath = os.path.join(savedir, filename + '.pt')
    print(f"Loading model from: {filepath}")
    
    model = load_escape_checkpoint_2d(filepath)
    mu_weight, s_weight = load_normalization_params_2d(suffix)
    
    model.to(device)
    model.eval()
    return model, mu_weight, s_weight

# Define the 2D model architecture
class EscapeModel2D(nn.Module):
    def __init__(self):
        super(EscapeModel2D, self).__init__()
        self.dim = 2  # 2D input
        self.hid_size = 512
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

def load_mc_data_2d(filename):
    """
    Load and process Monte Carlo data for comparison
    """
    filepath = os.path.join(root_data, filename)
    try:
        data = np.load(filepath, allow_pickle=True).item()
        
        # Extract information
        sampled_batches = data['sampled_data']
        time_points = data['times']
        grid_info = data.get('grid_info', {})
        
        # Concatenate all flags
        all_flags = np.concatenate([b['flags'] for b in sampled_batches], axis=1)
        
        num_points = grid_info.get('num_points', int(np.sqrt(all_flags.shape[1])))
        num_realizations = grid_info.get('num_realizations', 1)
        num_grid_points = grid_info.get('num_grid_points', all_flags.shape[1])
        
        print(f"Loaded MC data from: {filename}")
        print(f"Time points: {time_points}")
        print(f"Grid points: {num_grid_points} ({num_points}×{num_points})")
        print(f"Realizations per point: {num_realizations}")
        print(f"Total samples: {all_flags.shape[1]}")
        
        # Calculate escape probabilities
        if num_realizations > 1:
            # Reshape to (num_grid_points, num_realizations) and average
            flags_reshaped = all_flags.reshape(num_grid_points, num_realizations)
            escape_prob_flat = 1 - np.mean(flags_reshaped, axis=1)
        else:
            # Single realization per point
            escape_prob_flat = 1 - all_flags.flatten()
        
        # Reshape to grid
        escape_prob_grid = escape_prob_flat.reshape(num_points, num_points)
        
        return escape_prob_grid, num_points, data
        
    except FileNotFoundError:
        print(f"MC data file {filepath} not found.")
        return None, None, None

# Load both models and normalization parameters
print("Loading lower range model...")
Escape_lower, mu_weight_lower, s_weight_lower = load_escape_model_2d('_lower')

print("Loading upper range model...")
Escape_upper, mu_weight_upper, s_weight_upper = load_escape_model_2d('_upper')

print(f"Loaded lower model with normalization - Mean: {mu_weight_lower}, Std: {s_weight_lower}")
print(f"Loaded upper model with normalization - Mean: {mu_weight_upper}, Std: {s_weight_upper}")

# Convert normalization parameters to tensors
mu_weight_lower = torch.tensor(mu_weight_lower, dtype=torch.float32, device=device)
s_weight_lower = torch.tensor(s_weight_lower, dtype=torch.float32, device=device)
mu_weight_upper = torch.tensor(mu_weight_upper, dtype=torch.float32, device=device)
s_weight_upper = torch.tensor(s_weight_upper, dtype=torch.float32, device=device)

# Parameter ranges for 2D evaluation
x1_min, x1_max = -np.pi, np.pi   # Range for first dimension
x2_min, x2_max = 0, 2            # Range for second dimension
num_points = 101                 # Number of points per axis

# Generate the mesh grid for 2D
x1_vals = np.linspace(x1_min, x1_max, num_points)
x2_vals = np.linspace(x2_min, x2_max, num_points)

# Create meshgrid
X1, X2 = np.meshgrid(x1_vals, x2_vals, indexing='ij')

# Flatten and stack into a 2D array
coordinates_2d = np.stack([X1.ravel(), X2.ravel()], axis=1)

def predict_escape_probability_2d(coordinates):
    """
    Predict escape probability for 2D coordinates using range-based models
    """
    # Get the second dimension values
    x2_values = coordinates[:, 1]
    
    # Create masks for different ranges
    mask_lower = x2_values < 0.8
    mask_middle = (x2_values >= 0.8) & (x2_values <= 1.2)
    mask_upper = x2_values > 1.2
    
    # Initialize with staying probabilities (1 = staying, 0 = exited)
    stay_probabilities = np.ones(len(coordinates))
    
    print(f"Points in lower range (< 0.8): {np.sum(mask_lower)}")
    print(f"Points in middle range (0.8-1.2): {np.sum(mask_middle)}")
    print(f"Points in upper range (> 1.2): {np.sum(mask_upper)}")
    
    # Process lower range (< 0.8) with lower model
    if np.any(mask_lower):
        coords_lower = coordinates[mask_lower]
        test_lower = (coords_lower - mu_weight_lower.cpu().numpy()) / s_weight_lower.cpu().numpy()
        test_tensor_lower = torch.tensor(test_lower, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            probs_lower = Escape_lower(test_tensor_lower).cpu().numpy().flatten()
        stay_probabilities[mask_lower] = probs_lower
    
    # Middle range (0.8 <= x2 <= 1.2) - stay_probability = 1 (already initialized)
    # No processing needed for middle range
    
    # Process upper range (> 1.2) with upper model
    if np.any(mask_upper):
        coords_upper = coordinates[mask_upper]
        test_upper = (coords_upper - mu_weight_upper.cpu().numpy()) / s_weight_upper.cpu().numpy()
        test_tensor_upper = torch.tensor(test_upper, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            probs_upper = Escape_upper(test_tensor_upper).cpu().numpy().flatten()
        stay_probabilities[mask_upper] = probs_upper
    
    return stay_probabilities

# Compute NN escape probabilities for all points
print("Computing NN escape probabilities...")
stay_probabilities_flat = predict_escape_probability_2d(coordinates_2d)

# Convert staying probabilities to escape probabilities
escape_probabilities_flat = 1 - stay_probabilities_flat

# Reshape back to 2D grid
escape_probabilities_nn = escape_probabilities_flat.reshape(X1.shape)

print(f"NN Escape probability range: [{escape_probabilities_nn.min():.3f}, {escape_probabilities_nn.max():.3f}]")

##################################################################################
# Load Monte Carlo Data
##################################################################################

print("\n" + "="*60)
print("LOADING MONTE CARLO DATA FOR COMPARISON")
print("="*60)

# Try to load MC data with different naming conventions
mc_data_available = False
mc_data_101 = None
mc_data_31 = None

# Try new naming convention first (with number of realizations)

filename_101 = f'escape_2d_true_datanx101_nr{2000}.npy'
filename_31 = f'escape_2d_true_datanx31_nr{1000}.npy'


mc_escape_101, mc_num_points_101, mc_raw_101 = load_mc_data_2d(filename_101)
if mc_escape_101 is not None:
    mc_data_101 = (mc_escape_101, mc_num_points_101, filename_101)
    mc_data_available = True


    
mc_escape_31, mc_num_points_31, mc_raw_31 = load_mc_data_2d(filename_31)
if mc_escape_31 is not None:
    mc_data_31 = (mc_escape_31, mc_num_points_31, filename_31)



if mc_data_available:
    print("Monte Carlo data loaded successfully!")
else:
    print("Warning: No Monte Carlo data found. Only NN results will be shown.")

##################################################################################
# Plotting - Three plots in one row: NN, MC 101, MC 31
##################################################################################

# Create output directory
os.makedirs(figdir, exist_ok=True)

# Create three plots in one row with space for shared colorbar
fig = plt.figure(figsize=(22, 6))

# Create subplots with specific positioning to leave space for colorbar
gs = fig.add_gridspec(1, 3, left=0.05, right=0.85, wspace=0.3)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

levels = np.linspace(0, 1, 21)

# Plot 1: NN results
im_nn = axes[0].contourf(X1, X2, escape_probabilities_nn, levels=levels, 
                        cmap="RdBu_r", vmin=0, vmax=1)
x_ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
x_ticklabels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']
axes[0].set_xticks(x_ticks)
axes[0].set_xticklabels(x_ticklabels, fontsize=16)
axes[0].set_title('Our method', fontsize=20)
axes[0].set_xlabel('$x_1$', fontsize=18)
axes[0].set_ylabel('$x_2$', fontsize=18)
axes[0].tick_params(labelsize=16)



# Plot 2: MC results (nx=101)
if mc_data_101 is not None:
    mc_escape_101, mc_num_points_101, mc_filename_101 = mc_data_101
    
    # Create coordinate grid for MC data
    x1_mc_101 = np.linspace(x1_min, x1_max, mc_num_points_101)
    x2_mc_101 = np.linspace(x2_min, x2_max, mc_num_points_101)
    X1_mc_101, X2_mc_101 = np.meshgrid(x1_mc_101, x2_mc_101, indexing='ij')
    
    im_mc_101 = axes[1].contourf(X1_mc_101, X2_mc_101, mc_escape_101, 
                                levels=levels, cmap="RdBu_r", vmin=0, vmax=1)
    axes[1].set_title(f'MC: nx={mc_num_points_101}', fontsize=20)
    axes[1].set_xlabel('$x_1$', fontsize=18)
    axes[1].set_ylabel('$x_2$', fontsize=18)
    axes[1].tick_params(labelsize=16)
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(x_ticklabels, fontsize=16)      
    # Store the last plot for colorbar
    last_im = im_mc_101
else:
    axes[1].text(0.5, 0.5, 'MC data (nx=101)\nnot available', 
                ha='center', va='center', transform=axes[1].transAxes, fontsize=18)
    axes[1].set_title('MC: nx=101 (Not Available)', fontsize=20)
    axes[1].set_xlabel('$x_1$', fontsize=18)
    axes[1].set_ylabel('$x_2$', fontsize=18)
    axes[1].tick_params(labelsize=16)
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(x_ticklabels, fontsize=16)
    last_im = im_nn

# Plot 3: MC results (nx=31)
if mc_data_31 is not None:
    mc_escape_31, mc_num_points_31, mc_filename_31 = mc_data_31
    
    # Create coordinate grid for MC data
    x1_mc_31 = np.linspace(x1_min, x1_max, mc_num_points_31)
    x2_mc_31 = np.linspace(x2_min, x2_max, mc_num_points_31)
    X1_mc_31, X2_mc_31 = np.meshgrid(x1_mc_31, x2_mc_31, indexing='ij')
    
    im_mc_31 = axes[2].contourf(X1_mc_31, X2_mc_31, mc_escape_31, 
                               levels=levels, cmap="RdBu_r", vmin=0, vmax=1)
    axes[2].set_title(f'MC: nx={mc_num_points_31}', fontsize=20)
    axes[2].set_xlabel('$x_1$', fontsize=18)
    axes[2].set_ylabel('$x_2$', fontsize=18)
    axes[2].tick_params(labelsize=16)
    axes[2].set_xticks(x_ticks)
    axes[2].set_xticklabels(x_ticklabels, fontsize=16)
    # Store the last plot for colorbar
    last_im = im_mc_31
else:
    axes[2].text(0.5, 0.5, 'MC data (nx=31)\nnot available', 
                ha='center', va='center', transform=axes[2].transAxes, fontsize=18)
    axes[2].set_title('MC: nx=31 (Not Available)', fontsize=20)
    axes[2].set_xlabel('$x_1$', fontsize=18)
    axes[2].set_ylabel('$x_2$', fontsize=18)
    axes[2].tick_params(labelsize=16)
    axes[2].set_xticks(x_ticks)
    axes[2].set_xticklabels(x_ticklabels, fontsize=16)
# Add shared colorbar in the reserved space on the right
cbar_ax = fig.add_axes([0.87, 0.15, 0.01, 0.8])  # [left, bottom, width, height]
cbar = fig.colorbar(last_im, cax=cbar_ax)
cbar.set_label('Escape Probability', fontsize=18)
cbar.ax.tick_params(labelsize=16)

plt.savefig(os.path.join(figdir, '2d_escape_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n2D Escape probability analysis completed!")
print("Check the 'figures/' directory for output plots:")
print("  - 2d_escape_comparison.png: NN vs MC comparison (three plots in one row)")