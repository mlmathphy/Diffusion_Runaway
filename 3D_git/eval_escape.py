import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x_dim = 3
sde_dt = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_output = 'output/'   # where to save trained models
root_data = 'data/'       # where the datasets are

def load_escape_checkpoint(filepath):
    """
    Load checkpoint for 3D escape model
    """
    try:
        # First try with weights_only=False since we have numpy arrays in the checkpoint
        checkpoint = torch.load(filepath, weights_only=False, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Trying alternative loading method...")
        checkpoint = torch.load(filepath, map_location=device)
    
    model = EscapeModel().to(device)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load normalization parameters if available
    if 'normalization' in checkpoint:
        mu_weight = checkpoint['normalization']['mean']
        s_weight = checkpoint['normalization']['std']
        print("Loaded normalization parameters from checkpoint")
    else:
        print("Loading normalization parameters from separate file...")
        # Fallback to separate file
        try:
            with open(os.path.join(root_data, 'weight_mean_std.npy'), 'rb') as f:
                mu_weight = np.load(f)
                s_weight = np.load(f)
        except FileNotFoundError:
            print("Warning: No normalization file found. Using default values.")
            mu_weight = np.array([2.75, 0.0, 0.5])  # Rough estimates for (p_par, p_per, r)
            s_weight = np.array([1.5, 0.6, 0.3])
    
    return model, mu_weight, s_weight

def load_escape_model():
    """
    Load the trained 3D escape model
    """
    savedir = root_output
    filename = 'NN_3D_escape_model'  # Updated filename from training code
    
    filepath = os.path.join(savedir, filename + '.pt')
    print(f"Loading model from: {filepath}")
    
    model, mu_weight, s_weight = load_escape_checkpoint(filepath)
    model.to(device)
    model.eval()
    return model, mu_weight, s_weight

# Define the 3D model architecture (matching training code)
class EscapeModel(nn.Module):
    # def __init__(self):
    #     super(EscapeModel, self).__init__()
    #     self.dim = 3  # 3D: (p_par, p_per, r)
    #     self.hid_size = 512  # Match training code
    #     self.net = nn.Sequential(
    #         nn.Linear(self.dim, self.hid_size),
    #         nn.LeakyReLU(0.01),
    #         nn.BatchNorm1d(self.hid_size),
    #         nn.Dropout(0.3),
            
    #         nn.Linear(self.hid_size, self.hid_size),
    #         nn.LeakyReLU(0.01),
    #         nn.BatchNorm1d(self.hid_size),
    #         nn.Dropout(0.3),
            
    #         nn.Linear(self.hid_size, self.hid_size//2),
    #         nn.LeakyReLU(0.01),
    #         nn.BatchNorm1d(self.hid_size//2),
    #         nn.Dropout(0.2),
            
    #         nn.Linear(self.hid_size//2, self.hid_size//4),
    #         nn.LeakyReLU(0.01),
    #         nn.Dropout(0.1),
            
    #         nn.Linear(self.hid_size//4, 1),
    #         nn.Sigmoid()  # Add sigmoid back for evaluation
    #     )
    def __init__(self):
        super(EscapeModel, self).__init__()
        self.dim = 3
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

# Load the trained model and normalization parameters
Escape, mu_weight, s_weight = load_escape_model()
print(f"Loaded model with normalization - Mean: {mu_weight}, Std: {s_weight}")

# Parameter ranges
p_min, p_max = 0.5, 5  # Define the range for p
xi_min, xi_max = -1, 1
r_min, r_max = 0.8, 1  # Define the range for r
num_points = 31  # Number of points per axis

# Define the threshold for r
r_threshold = 0.81

# Generate the mesh grid for x1 and x2
p1 = np.linspace(p_min, p_max, num_points)
xi1 = np.linspace(xi_min, xi_max, num_points)
r1 = np.linspace(r_min, r_max, num_points)

# Create meshgrid in 'ij' indexing to preserve the input order
P, XI, R = np.meshgrid(p1, xi1, r1, indexing='ij')

# Convert to parallel and perpendicular components
P_par = P * XI
P_per = P * np.sqrt(1 - XI**2)

# Flatten and stack into a 2D array
coordinates_3d = np.stack([P_par.ravel(), P_per.ravel(), R.ravel()], axis=1)

# Create mask for points where r >= threshold
r_mask = coordinates_3d[:, 2] >= r_threshold

# Initialize probabilities array with ones (staying probability)
probabilities_flat = np.ones(len(coordinates_3d))

# Only process points where r >= threshold
if np.any(r_mask):
    print(f"Computing escape probabilities for {np.sum(r_mask)} points where r >= {r_threshold}...")
    
    # Extract coordinates where r >= threshold
    coordinates_above_threshold = coordinates_3d[r_mask]
    
    # Normalize using the same parameters as training
    test_normalized = (coordinates_above_threshold - mu_weight) / s_weight
    test_tensor = torch.tensor(test_normalized, dtype=torch.float32).to(device)
    
    # Predict probabilities in batches to handle memory efficiently
    batch_size = 10000
    batch_probabilities = []
    
    with torch.no_grad():
        for i in range(0, len(test_tensor), batch_size):
            batch = test_tensor[i:i+batch_size]
            batch_probs = Escape(batch).cpu().numpy()
            batch_probabilities.append(batch_probs)
    
    batch_probabilities = np.concatenate(batch_probabilities, axis=0).flatten()
    
    # Assign the computed probabilities to the corresponding positions
    probabilities_flat[r_mask] = batch_probabilities

print(f"Points below threshold (r < {r_threshold}): {np.sum(~r_mask)}")
print(f"Points above threshold (r >= {r_threshold}): {np.sum(r_mask)}")

# Reshape probabilities back to 3D grid

probabilities = probabilities_flat.reshape(P.shape)
r_max_mask = (R >= r_max)
probabilities[r_max_mask] = 0




##################################################################################
# plot 1: escape rate comparison
##################################################################################

# Create output directory
os.makedirs('figures', exist_ok=True)

try:
    # Load the saved Monte Carlo data
    data = np.load(os.path.join('data', 'escape_true_datanx101.npy'), allow_pickle=True).item()
    num_points_MC = 101  # Number of points per axis

    # Generate the mesh grid for Monte Carlo data
    p1_MC = np.linspace(p_min, p_max, num_points_MC)
    xi1_MC = np.linspace(xi_min, xi_max, num_points_MC)
    r1_MC = np.linspace(r_min, r_max, num_points_MC)

    sampled_batches = data['sampled_data']
    time_points = data['times']  # This is your t1 axis

    # Concatenate all flags across batches
    all_flags = np.concatenate([b['flags'] for b in sampled_batches], axis=1)  # shape (T, N)

    # Compute normalized stay curve over time
    true_stay_curve = np.mean(all_flags, axis=1)  # shape (T,)
    true_stay_curve = true_stay_curve / true_stay_curve[0]  # Normalize to 1 at t = 0
    print("True stay curve loaded successfully")
    print(f"True stay curve shape: {true_stay_curve.shape}")
    
    mc_data_available = True
    
except FileNotFoundError:
    print("Warning: Monte Carlo data file not found. Skipping MC comparison plots.")
    mc_data_available = False

##################################################################################
# plot 2: fixed t = 20, cross section (p, xi), (xi, r), (r,p)
##################################################################################

if mc_data_available:
    theta1 = np.arccos(np.clip(xi1, -1, 1))  # used for theta-angle axis
    theta1_MC = np.arccos(np.clip(xi1_MC, -1, 1))
    
    # Compute cross sections from NN (probabilities)
    prob_nn = 1 - probabilities[:, :, :]  # Convert staying prob to escape prob
    
    # Reshape MC data - take final time step for comparison
    prob_mc = 1 - all_flags[-1].reshape(num_points_MC, num_points_MC, num_points_MC)
    
    try:
        # Load additional MC data (nx = 31)
        data_31 = np.load(os.path.join('data', 'escape_true_datanx31.npy'), allow_pickle=True).item()
        sampled_batches_31 = data_31['sampled_data']
        num_points_MC31 = 31
        p1_MC31 = np.linspace(p_min, p_max, num_points_MC31)
        xi1_MC31 = np.linspace(xi_min, xi_max, num_points_MC31)
        r1_MC31 = np.linspace(r_min, r_max, num_points_MC31)
        theta1_MC31 = np.arccos(np.clip(xi1_MC31, -1, 1))
        all_flags_31 = np.concatenate([b['flags'] for b in sampled_batches_31], axis=1)
        prob_mc_31 = 1 - all_flags_31[-1].reshape(num_points_MC31, num_points_MC31, num_points_MC31)
        mc31_available = True
    except FileNotFoundError:
        print("Warning: MC data (nx=31) not found. Using only nx=101 comparison.")
        mc31_available = False

    # Determine number of columns based on available data
    n_cols = 3 if mc31_available else 2
    
    fig, axs = plt.subplots(2, n_cols, figsize=(6*n_cols, 8), constrained_layout=True)
    if n_cols == 2:
        axs = axs.reshape(3, 2)

   # ---------------- Row 2: (theta, r) ----------------
    # NN
    im2 = axs[0, 0].contourf(theta1, r1, np.mean(prob_nn, axis=0).T, levels=20, cmap='plasma', vmin=0, vmax=1)
    axs[0, 0].set_title(r'NN: $\theta$ vs $r$', fontsize=16)
    axs[0, 0].set_xlabel(r'$\theta$', fontsize=16)
    axs[0, 0].set_ylabel('r', fontsize=16)
    axs[0, 0].set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    axs[0, 0].set_xticklabels(['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=14)
    axs[0, 0].tick_params(labelsize=14)
    fig.colorbar(im2, ax=axs[0, 0]).ax.tick_params(labelsize=12)

    # MC (nx=101)
    im3 = axs[0, 1].contourf(theta1_MC, r1_MC, np.mean(prob_mc, axis=0).T, levels=20, cmap='plasma', vmin=0, vmax=1)
    axs[0, 1].set_title(r'MC ($n_x$=101): $\theta$ vs $r$', fontsize=16)
    axs[0, 1].set_xlabel(r'$\theta$', fontsize=16)
    axs[0, 1].set_ylabel('r', fontsize=16)
    axs[0, 1].set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    axs[0, 1].set_xticklabels(['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=14)
    axs[0, 1].tick_params(labelsize=14)
    fig.colorbar(im3, ax=axs[0, 1]).ax.tick_params(labelsize=12)

    # MC (nx=31) if available
    if mc31_available:
        im_extra1 = axs[0, 2].contourf(theta1_MC31, r1_MC31, np.mean(prob_mc_31, axis=0).T, levels=20, cmap='plasma', vmin=0, vmax=1)
        axs[0, 2].set_title(r'MC ($n_x$=31): $\theta$ vs $r$', fontsize=16)
        axs[0, 2].set_xlabel(r'$\theta$', fontsize=16)
        axs[0, 2].set_ylabel('r', fontsize=16)
        axs[0, 2].set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        axs[0, 2].set_xticklabels(['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=14)
        axs[0, 2].tick_params(labelsize=14)
        fig.colorbar(im_extra1, ax=axs[0, 2]).ax.tick_params(labelsize=12)

    # ---------------- Row 3: (r, p) ----------------
    # NN
    im4 = axs[1, 0].contourf(r1, p1, np.mean(prob_nn, axis=1), levels=20, cmap='plasma', vmin=0, vmax=1)
    axs[1, 0].set_title('NN: $r$ vs $p$', fontsize=16)
    axs[1, 0].set_xlabel('r', fontsize=16)
    axs[1, 0].set_ylabel('p', fontsize=16)
    axs[1, 0].tick_params(labelsize=14)
    fig.colorbar(im4, ax=axs[1, 0]).ax.tick_params(labelsize=12)

    # MC (nx=101)
    im5 = axs[1, 1].contourf(r1_MC, p1_MC, np.mean(prob_mc, axis=1), levels=20, cmap='plasma', vmin=0, vmax=1)
    axs[1, 1].set_title('MC ($n_x$=101): $r$ vs $p$', fontsize=16)
    axs[1, 1].set_xlabel('r', fontsize=16)
    axs[1, 1].set_ylabel('p', fontsize=16)
    axs[1, 1].tick_params(labelsize=14)
    fig.colorbar(im5, ax=axs[1, 1]).ax.tick_params(labelsize=12)

    # MC (nx=31) if available
    if mc31_available:
        im_extra2 = axs[1, 2].contourf(r1_MC31, p1_MC31, np.mean(prob_mc_31, axis=1), levels=20, cmap='plasma', vmin=0, vmax=1)
        axs[1, 2].set_title('MC ($n_x$=31): $r$ vs $p$', fontsize=16)
        axs[1, 2].set_xlabel('r', fontsize=16)
        axs[1, 2].set_ylabel('p', fontsize=16)
        axs[1, 2].tick_params(labelsize=14)
        fig.colorbar(im_extra2, ax=axs[1, 2]).ax.tick_params(labelsize=12)

    # Save figure
    plt.suptitle(f'Escape Probability Comparison', fontsize=20)
    plt.savefig('figures/escape_comparison_cross_section.png', dpi=300, bbox_inches='tight')
    plt.show()

else:
    print("Skipping cross-section plots due to missing Monte Carlo data.")
    
print("Analysis completed!")
print("Check the 'figures/' directory for output plots.")