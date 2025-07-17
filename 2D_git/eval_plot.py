import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.io import loadmat
from scipy.stats import norm
from tqdm import tqdm

from matplotlib.lines import Line2D
sde_dt = 0.05  # Using the 2D time step
sde_T = 1  # Using your 2D simulation time
# 1. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_dim = 2  # 2D problem

datadir = os.path.join(f'data_2d')
figdir = os.path.join(f'fig_2d')

# 2D domain parameters from your data generation code
x_min, x_max = -np.pi, np.pi  # x1 domain
y_min, y_max = 0.0, 2.0       # x2 domain

# Physical parameters
Pe = 5.
epsilon = 0.0
n = 2

print(datadir)

def velocity_field(x1, x2):
    """
    Compute the autonomous velocity field v = (v_x1, v_x2) at position (x1, x2)
    """
    f_t = 0.0
    
    v_x1 = -np.pi * np.cos(np.pi * x2) * (np.sin(n * x1) + epsilon * f_t * np.cos(n * x1))
    v_x2 = n * np.sin(np.pi * x2) * (np.cos(n * x1) - epsilon * f_t * np.sin(n * x1))
    
    return v_x1, v_x2

def apply_periodic_bc(x1, x2):
    """
    Apply periodic boundary conditions in x1 direction
    and reflective boundary conditions in x2 direction
    """
    # Periodic in x1: [-π, π]
    x1 = ((x1 - x_min) % (x_max - x_min)) + x_min
    
    return x1, x2

def check_exit_condition(x1, x2):
    """
    Check if particles have exited the domain by reaching x2=0 or x2=1 boundaries
    Returns flag array: 1 for still inside domain, 0 for exited
    """
    flag = np.ones(len(x1))
    
    # Exit condition: particle reaches top (x2 >= 1) or bottom (x2 <= 0) boundary
    exited = (x2 >= y_max) | (x2 <= y_min)
    flag[exited] = 0
    
    return flag

def SDE_2d_advection_diffusion(T, dt, dim, Nsample, idx_initial):
    """
    2D Advection-Diffusion SDE solver for evaluation
    dx1 = Pe * v_x1(x1, x2, t) * dt + dW1
    dx2 = Pe * v_x2(x1, x2, t) * dt + dW2
    """
    
    # SDE parameters
    mc_dt = 0.0005    # Fine time step for numerical integration
    t_needed = int(dt / mc_dt)
    Nt = int(np.floor(T / mc_dt) + 1)
    N_snap = int(np.floor(T / dt) + 1)
    
    # Initial conditions based on idx_initial
    if idx_initial == 1:
        # Random initial positions
        x1_ini = np.random.uniform(low=x_min, high=x_max, size=Nsample)
        x2_ini = np.random.uniform(low=y_min, high=y_max, size=Nsample)
    elif idx_initial == 2:
        # Fixed positions
        x1_ini = np.pi/2 * np.ones(Nsample)  # Some fixed x1
        x2_ini = 1 * np.ones(Nsample)      # Near top boundary
    elif idx_initial == 3:
        # Center positions
        x1_ini = np.zeros(Nsample)           # Center in x1
        x2_ini = 1 * np.ones(Nsample)     # Center in x2
    elif idx_initial == 4:
        # Center positions
        x1_ini = -1.0 * np.pi/10 * np.ones(Nsample)           # Center in x1
        x2_ini = 1 * np.ones(Nsample)     # Center in x2    
    # Current positions
    x1_end = np.copy(x1_ini)
    x2_end = np.copy(x2_ini)
    flag = np.ones(Nsample)
    n_flag = Nsample
    
    # Storage arrays
    sampled_trajectory_x1 = np.zeros((Nsample, N_snap))
    sampled_trajectory_x2 = np.zeros((Nsample, N_snap))
    exit_trajectory = np.zeros((Nsample, N_snap))
    
    # Initialize storage
    sampled_trajectory_x1[:, 0] = x1_end
    sampled_trajectory_x2[:, 0] = x2_end
    exit_trajectory[:, 0] = flag
    
    sample_idx = 1
    
    print("Starting SDE integration...")
    for ii in range(Nt):
        if n_flag <= 0:
            break
            
        if ii % 50000 == 0:
            print(f"Step {ii}/{Nt}, active particles: {n_flag}")
        
        # Get active particles
        idx_flag = np.where(flag == 1)[0]
        n_active = len(idx_flag)
        
        if n_active == 0:
            break
            
        # Compute velocity field for active particles
        v_x1, v_x2 = velocity_field(x1_end[idx_flag], x2_end[idx_flag])
        
        # Generate random increments
        dW1 = np.random.normal(0, np.sqrt(mc_dt), n_active)
        dW2 = np.random.normal(0, np.sqrt(mc_dt), n_active)
        
        # Update positions using SDE
        x1_end[idx_flag] += Pe * v_x1 * mc_dt + dW1
        x2_end[idx_flag] += Pe * v_x2 * mc_dt + dW2
        
        # Apply boundary conditions
        x1_end[idx_flag], x2_end[idx_flag] = apply_periodic_bc(x1_end[idx_flag], x2_end[idx_flag])
        
        # Check if particles are still valid
        flag = check_exit_condition(x1_end, x2_end)
        n_flag = np.sum(flag)
        
        # Store snapshots
        if (ii + 1) % t_needed == 0 and sample_idx < N_snap:
            sampled_trajectory_x1[:, sample_idx] = x1_end
            sampled_trajectory_x2[:, sample_idx] = x2_end
            exit_trajectory[:, sample_idx] = flag
            sample_idx += 1
    
    # Combine trajectories
    x_ini = np.column_stack([x1_ini, x2_ini])
    sampled_trajectory = np.stack([sampled_trajectory_x1, sampled_trajectory_x2], axis=-1)
    
    return x_ini, sampled_trajectory, exit_trajectory


# 3. Define your NN classes (same as training)
class FN_Net(nn.Module):
    def __init__(self, input_dim, output_dim, hid_size):
        super(FN_Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_size = hid_size
        
        self.input = nn.Linear(self.input_dim, self.hid_size)
        self.fc1 = nn.Linear(self.hid_size, self.hid_size)
        self.output = nn.Linear(self.hid_size, self.output_dim)

    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.fc1(x))
        x = self.output(x)
        return x

# 4. Load trained FN model
FN = FN_Net(x_dim*2, x_dim, 128).to(device)  # Using your exact architecture
FN.load_state_dict(torch.load(os.path.join(datadir, 'FN.pth'), map_location=device, weights_only=True))
FN.eval()

def load_weight_checkpoint(filepath):
    """
    Load checkpoint for escape model
    """
    checkpoint = torch.load(filepath, weights_only=True)
    model = EscapeModel2D().to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_weight_model(model_suffix=''):
    """
    Load escape model with optional suffix
    """
    savedir = datadir
    filename = f'/NN_time_weight_2d{model_suffix}'

    print(savedir + filename + '.pt')
    model = load_weight_checkpoint(savedir + filename + '.pt')
    model.to(device)
    model.eval()
    return model

def load_normalization_params(suffix=''):
    """
    Load normalization parameters with optional suffix
    """
    with open(datadir + f'/weight_mean_std_2d{suffix}.npy', 'rb') as f:
        mu_weight = np.load(f)
        s_weight = np.load(f)
    return mu_weight, s_weight

# Define escape model (same as your training code)
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

# Load both models and normalization parameters
Escape_lower = load_weight_model('_lower')
Escape_upper = load_weight_model('_upper')

mu_weight_lower, s_weight_lower = load_normalization_params('_lower')
mu_weight_upper, s_weight_upper = load_normalization_params('_upper')

# Convert to tensors
mu_weight_lower = torch.tensor(mu_weight_lower, dtype=torch.float32, device=device)
s_weight_lower = torch.tensor(s_weight_lower, dtype=torch.float32, device=device)
mu_weight_upper = torch.tensor(mu_weight_upper, dtype=torch.float32, device=device)
s_weight_upper = torch.tensor(s_weight_upper, dtype=torch.float32, device=device)

#----------------------------------------------------------------------------------------------------
#  Main
#-----------------------------------------------------------------------------------------------------
# Load data info
data_inf = torch.load(os.path.join(datadir, 'data_inf.pt'), weights_only=False)
xTrain_mean = data_inf['xTrain_mean'].to(device)
xTrain_std = data_inf['xTrain_std'].to(device)
yTrain_mean = data_inf['yTrain_mean'].to(device)
yTrain_std = data_inf['yTrain_std'].to(device)
diff_scale = data_inf['diff_scale']
print("diff_scale:", diff_scale)

print(xTrain_mean)

Nsample = 100000  # Reduced for 2D

idx_initial = 2
ode_path_file = os.path.join(datadir, f'ode_path_2d_{idx_initial}.npy')

ode_time_steps = int(np.floor(sde_T/sde_dt))
print('ode_time_steps is:', ode_time_steps)

# Check if file exists
if os.path.exists(ode_path_file):
    print(f"Loading existing 2D path from {ode_path_file}")
    loaded = np.load(ode_path_file, allow_pickle=True).item()
    true_init = loaded['true_init']
    sampled_trajectory = loaded['sampled_trajectory']
    exit_trajectory = loaded['exit_trajectory']
else:
    print("File not found. Running SDE_2d_advection_diffusion to generate data...")
    true_init, sampled_trajectory, exit_trajectory = SDE_2d_advection_diffusion(sde_T, sde_dt, x_dim, Nsample, idx_initial)
    save_data = {
        'true_init': true_init,
        'sampled_trajectory': sampled_trajectory,
        'exit_trajectory': exit_trajectory
    }
    # np.save(ode_path_file, save_data)
    print(f"Saved generated 2D path to {ode_path_file}")

print("sampled_trajectory shape is: ", sampled_trajectory.shape)

# Process final results
exit_final = exit_trajectory[:, -1]
print(f"Number of particles that exited: {np.sum(exit_final == 0)}")
print(f"Number of particles still active: {np.sum(exit_final == 1)}")

y_true_transform = sampled_trajectory[:, -1, :][exit_final == 1]  # Final positions of active particles
print('y_true_transform shape:', y_true_transform.shape)

# Choose comparison time step
comparison_step = 5  # Change this to compare at different time steps (0 to ode_time_steps-1)
print(f"Will compare distributions at time step {comparison_step} (out of {ode_time_steps-1})")

# Neural network prediction
x_pred_new = torch.tensor(true_init, dtype=torch.float32, device=device)

# Store predicted trajectories for comparison
pred_trajectory = []
pred_trajectory.append(x_pred_new.cpu().detach().numpy().copy())

print("Starting neural network simulation...")
print("Initial particles:", x_pred_new.size(0))

def predict_escape_probability(x_pred_new):
    """
    Predict escape probability based on the second dimension range
    """
    if x_pred_new.size(0) == 0:
        return torch.tensor([], device=device)
    
    # Get the second dimension values
    x2_values = x_pred_new[:, 1]
    
    # Create masks for different ranges
    mask_lower = x2_values < 0.8
    mask_middle = (x2_values >= 0.8) & (x2_values <= 1.2)
    mask_upper = x2_values > 1.2
    
    # Initialize weight prediction tensor
    weight_pred = torch.ones(x_pred_new.size(0), 1, device=device)
    
    # Process lower range (< 0.8)
    if mask_lower.any():
        x_lower = x_pred_new[mask_lower]
        test_lower = (x_lower - mu_weight_lower) / s_weight_lower
        weight_pred[mask_lower] = Escape_lower(test_lower)
    
    # Process middle range (0.8 <= x <= 1.2) - weight_pred = 1 (already initialized)
    # No processing needed for middle range as it's already set to 1
    
    # Process upper range (> 1.2)
    if mask_upper.any():
        x_upper = x_pred_new[mask_upper]
        test_upper = (x_upper - mu_weight_upper) / s_weight_upper
        weight_pred[mask_upper] = Escape_upper(test_upper)
    
    return weight_pred

for jj in range(ode_time_steps):
    if x_pred_new.size(0) == 0:
        print(f"No particles remaining at step {jj}")
        break
        
    # Escape probability prediction using range-based models
    weight_pred = predict_escape_probability(x_pred_new)

    # Random sampling for escape
    random_values = torch.rand(len(weight_pred), device=device)
    mask_to_keep = weight_pred.flatten() > random_values.flatten()

    # Neural network flow prediction
    Npath_pred0 = x_pred_new.size(0)
    noise_input = torch.randn(Npath_pred0, x_dim).to(device, dtype=torch.float32)
    nn_input = torch.hstack((x_pred_new, noise_input))
    
    prediction = FN((nn_input - xTrain_mean) / xTrain_std) * yTrain_std + yTrain_mean
    prediction = prediction / torch.tensor(diff_scale, device=device, dtype=torch.float32) + x_pred_new

    # Apply escape mask
    prediction = prediction[mask_to_keep, :]
    
    # Apply periodic boundary conditions for x1 (periodic in [-π, π])
    prediction[:, 0] = ((prediction[:, 0] - x_min) % (x_max - x_min)) + x_min
    
    # Check if particles are still in valid x2 domain (exit if outside [0, 1])
    keep_prediction_flag = (prediction[:, 1] >= y_min) & (prediction[:, 1] <= y_max)
    
    prediction = prediction[keep_prediction_flag, :]
    x_pred_new = prediction.clone()
    
    # Store trajectory at each step
    if x_pred_new.size(0) > 0:
        pred_trajectory.append(x_pred_new.cpu().detach().numpy().copy())
    else:
        pred_trajectory.append(np.empty((0, 2)))
    
    # Progress reporting
    if jj % 10 == 0:
        print(f"Step {jj}/{ode_time_steps}, remaining particles: {x_pred_new.size(0)}")

# Set larger font sizes globally
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# Get data at comparison step
if comparison_step < len(pred_trajectory) and comparison_step < sampled_trajectory.shape[1]:
    # True data at comparison step
    true_at_step = sampled_trajectory[:, comparison_step, :]
    exit_at_step = exit_trajectory[:, comparison_step]
    y_true_step = true_at_step[exit_at_step == 1]  # Only active particles
    
    # Predicted data at comparison step
    y_pred_step = pred_trajectory[comparison_step]
    
    Npath_true = y_true_step.shape[0]
    Npath_pred = y_pred_step.shape[0]
    
    print(f"\nComparison at time step {comparison_step}:")
    print(f"True active particles: {Npath_true}")
    print(f"Predicted particles: {Npath_pred}")
    
    
    if y_pred_step.shape[0] > 0:
        print(f"Prediction x1 range: [{np.min(y_pred_step[:, 0]):.3f}, {np.max(y_pred_step[:, 0]):.3f}]")
        print(f"Prediction x2 range: [{np.min(y_pred_step[:, 1]):.3f}, {np.max(y_pred_step[:, 1]):.3f}]")
        
        # Define x1 ticks and labels for pi-based axes
        x_ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        x_ticklabels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']
        
        # 2D Visualization at comparison step using contour plots
        plt.figure(figsize=(10, 5))
        
        # True distribution - contour plot
        plt.subplot(1, 2, 1)
        # Create 2D histogram for contour
        deltaX = (max(y_true_step[:, 0]) - min(y_true_step[:, 0]))/19
        deltaY = (max(y_true_step[:, 1]) - min(y_true_step[:, 1]))/19
        hist_true, x_edges, y_edges = np.histogram2d(y_true_step[:, 0], y_true_step[:, 1], bins=20, 
                                                    range=[[x_min-deltaX, x_max+deltaX], [y_min-deltaY, y_max+deltaY]])
        X_true, Y_true = np.meshgrid(x_edges[:-1], y_edges[:-1])
        
        # Create contour plot
        custom_levels = np.linspace(np.min(hist_true), np.max(hist_true), 10)
        plt.contourf(X_true, Y_true, hist_true.T, levels=custom_levels, cmap='coolwarm', alpha=1)
        
        plt.title(f'True Distribution ({Npath_true} particles)', fontsize=16)
        plt.xlabel('$x_1$', fontsize=14)
        plt.ylabel('$x_2$', fontsize=14)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(x_ticks, x_ticklabels, fontsize=12)
        plt.yticks(fontsize=12)
        
        # Predicted distribution - contour plot
        plt.subplot(1, 2, 2)
        # Create 2D histogram for contour
        hist_pred, x_edges, y_edges = np.histogram2d(y_pred_step[:, 0], y_pred_step[:, 1], bins=20,
                                                    range=[[x_min-deltaX, x_max+deltaX], [y_min-deltaY, y_max+deltaY]])
        X_pred, Y_pred = np.meshgrid(x_edges[:-1], y_edges[:-1])
        
        # Create contour plot
        hist_pred = np.maximum(hist_pred, np.min(hist_true))
        hist_pred = np.minimum(hist_pred, np.max(hist_true))
        plt.contourf(X_pred, Y_pred, hist_pred.T, levels=custom_levels, cmap='coolwarm', alpha=1)
        
        plt.title(f'Predicted Distribution ({Npath_pred} particles)', fontsize=16)
        plt.xlabel('$x_1$', fontsize=14)
        plt.ylabel('$x_2$', fontsize=14)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(x_ticks, x_ticklabels, fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, f'comparison_2d_step_{comparison_step}_ini_{idx_initial}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional 1D marginal distributions using KDE curves
        plt.figure(figsize=(10, 5))
        
        # x1 marginal
        plt.subplot(1, 2, 1)
        
        # Calculate histogram counts for scaling KDE
        hist_true_x1, bin_edges_x1 = np.histogram(y_true_step[:, 0], bins=50)
        hist_pred_x1, _ = np.histogram(y_pred_step[:, 0], bins=bin_edges_x1)
        
        # Calculate KDE
        kde_true_x1 = gaussian_kde(y_true_step[:, 0])
        kde_pred_x1 = gaussian_kde(y_pred_step[:, 0])
        
        # Create x-axis for plotting
        x1_range = np.linspace(np.min([y_true_step[:, 0].min(), y_pred_step[:, 0].min()]),
                            np.max([y_true_step[:, 0].max(), y_pred_step[:, 0].max()]), 200)
        
        # Scale KDE to match count scale
        kde_true_x1_scaled = kde_true_x1(x1_range) * len(y_true_step[:, 0]) * (bin_edges_x1[1] - bin_edges_x1[0])
        kde_pred_x1_scaled = kde_pred_x1(x1_range) * len(y_pred_step[:, 0]) * (bin_edges_x1[1] - bin_edges_x1[0])
        
        plt.plot(x1_range, kde_true_x1_scaled, label='Ground-Truth', color='blue', linewidth=2)
        plt.plot(x1_range, kde_pred_x1_scaled, label='Our method', color='red', linewidth=2)
        plt.xlabel('$x_1$', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xlim([-np.pi,np.pi])
        plt.title('$x_1$ Marginal Distribution', fontsize=16)
        plt.legend(fontsize=12)
        plt.xticks(x_ticks, x_ticklabels, fontsize=12)
        plt.yticks(fontsize=12)
        
        # x2 marginal
        plt.subplot(1, 2, 2)
        
        # Calculate histogram counts for scaling KDE
        hist_true_x2, bin_edges_x2 = np.histogram(y_true_step[:, 1], bins=50)
        hist_pred_x2, _ = np.histogram(y_pred_step[:, 1], bins=bin_edges_x2)
        
        # Calculate KDE
        kde_true_x2 = gaussian_kde(y_true_step[:, 1])
        kde_pred_x2 = gaussian_kde(y_pred_step[:, 1])
        
        # Create x-axis for plotting
        x2_range = np.linspace(np.min([y_true_step[:, 1].min(), y_pred_step[:, 1].min()]),
                            np.max([y_true_step[:, 1].max(), y_pred_step[:, 1].max()]), 200)
        
        # Scale KDE to match count scale
        kde_true_x2_scaled = kde_true_x2(x2_range) * len(y_true_step[:, 1]) * (bin_edges_x2[1] - bin_edges_x2[0])
        kde_pred_x2_scaled = kde_pred_x2(x2_range) * len(y_pred_step[:, 1]) * (bin_edges_x2[1] - bin_edges_x2[0])
        
        plt.plot(x2_range, kde_true_x2_scaled, label='Ground-Truth', color='blue', linewidth=2)
        plt.plot(x2_range, kde_pred_x2_scaled, label='Our method', color='red', linewidth=2)
        plt.xlabel('$x_2$', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xlim([0,2])
        plt.title('$x_2$ Marginal Distribution', fontsize=16)
        plt.legend(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, f'marginal_distributions_2d_step_{comparison_step}_ini_{idx_initial}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        print("No predicted particles to plot at this time step")
        
else:
    print(f"Invalid comparison step {comparison_step}. Available steps: 0 to {min(len(pred_trajectory)-1, sampled_trajectory.shape[1]-1)}")

# Also show final results
exit_final = exit_trajectory[:, -1]
print(f"\nFinal Results:")
print(f"Particles that exited: {np.sum(exit_final == 0)}")
print(f"Particles still active: {np.sum(exit_final == 1)}")

if len(pred_trajectory) > 0:
    final_pred = pred_trajectory[-1]
    print(f"Final predicted particles: {final_pred.shape[0]}")
else:
    print("No final predicted particles")

print(f"\nSimulation complete. Results saved in {figdir}")