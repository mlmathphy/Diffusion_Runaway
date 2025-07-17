import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Setup
x_dim = 3  # Now 3D: (p, xi, r)
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
            mu_weight = np.array([2.75, 0.0, 0.5])  # Rough estimates for (p, xi, r)
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
    def __init__(self):
        super(EscapeModel, self).__init__()
        self.dim = 3  # 3D: (p, xi, r)
        self.hid_size = 512  # Match training code
        self.net = nn.Sequential(
            nn.Linear(self.dim, self.hid_size),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(self.hid_size),
            nn.Dropout(0.3),
            
            nn.Linear(self.hid_size, self.hid_size),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(self.hid_size),
            nn.Dropout(0.3),
            
            nn.Linear(self.hid_size, self.hid_size//2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(self.hid_size//2),
            nn.Dropout(0.2),
            
            nn.Linear(self.hid_size//2, self.hid_size//4),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            
            nn.Linear(self.hid_size//4, 1),
            nn.Sigmoid()  # Add sigmoid back for evaluation
        )
    
    def forward(self, x):
        return self.net(x)

# Load the trained model and normalization parameters
Escape, mu_weight, s_weight = load_escape_model()
print(f"Loaded model with normalization - Mean: {mu_weight}, Std: {s_weight}")

# Define parameter ranges for 3D space
p_min, p_max = 0.5, 8.0    # Momentum range
xi_min, xi_max = -1.0, 1.0  # Pitch angle cosine range  
r_min, r_max = 0.0, 1.0     # Radial position range

num_points = 50  # Increased resolution for better plots

# Generate the mesh grid for 3D space
p1 = np.linspace(p_min, p_max, num_points)
xi1 = np.linspace(xi_min, xi_max, num_points)
r1 = np.linspace(r_min, r_max, num_points)

# Create meshgrid in 'ij' indexing to preserve the input order
P, XI, R = np.meshgrid(p1, xi1, r1, indexing='ij')

P_per = P * XI
P_par = P * np.sqrt(1 - XI**2)

# Flatten and stack into a 2D array (3D coordinates)
coordinates_3d = np.stack([P_per.ravel(), P_par.ravel(), R.ravel()], axis=1)

# Normalize using the same parameters as training
test_normalized = (coordinates_3d - mu_weight) / s_weight
test_tensor = torch.tensor(test_normalized, dtype=torch.float32).to(device)

# Predict probabilities in batches to handle memory efficiently
batch_size = 10000
probabilities_flat = []

print("Computing escape probabilities...")
with torch.no_grad():
    for i in range(0, len(test_tensor), batch_size):
        batch = test_tensor[i:i+batch_size]
        batch_probs = Escape(batch).cpu().numpy()
        probabilities_flat.append(batch_probs)

probabilities_flat = np.concatenate(probabilities_flat, axis=0)
probabilities = probabilities_flat.reshape(P.shape)

print(f"Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
print(f"Mean escape probability: {1 - probabilities.mean():.3f}")

# Create output directory
os.makedirs('figures', exist_ok=True)

# Plot 1: Escape probability of (p, r) - averaged over xi
print("Creating (p, r) plot...")
prob_pr = np.mean(probabilities, axis=1)  # Average over xi dimension

fig, ax = plt.subplots(figsize=(10, 8))
c = ax.contourf(p1, r1, (1 - prob_pr).T, levels=30, cmap='plasma')
cbar = fig.colorbar(c, ax=ax)
cbar.set_label('Escape Probability', fontsize=14)
cbar.ax.tick_params(labelsize=12)

ax.set_xlabel('Momentum (p)', fontsize=16)
ax.set_ylabel('Radial Position (r)', fontsize=16)
ax.set_title('Escape Probability vs (p, r)', fontsize=16)
ax.tick_params(axis='both', labelsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/probability_p_vs_r.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Escape probability of (xi, r) - averaged over p
print("Creating (xi, r) plot...")
prob_xir = np.mean(probabilities, axis=0)  # Average over p dimension

fig, ax = plt.subplots(figsize=(10, 8))

# Convert xi to theta (pitch angle)
theta1 = np.arccos(np.clip(xi1, -1, 1))  # Ensure xi is in valid range for arccos

c = ax.contourf(theta1, r1, (1 - prob_xir).T, levels=30, cmap='plasma')
cbar = fig.colorbar(c, ax=ax)
cbar.set_label('Escape Probability', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Set theta axis labels
ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
ax.set_xticklabels(['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=16)
ax.set_xlabel(r'Pitch Angle $\theta$', fontsize=16)
ax.set_ylabel('Radial Position (r)', fontsize=16)
ax.set_title(r'Escape Probability vs ($\theta$, r)', fontsize=16)
ax.tick_params(axis='both', labelsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/probability_xi_vs_r.png', dpi=300, bbox_inches='tight')
plt.close()

# Additional analysis: Plot escape probability vs each individual parameter
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot vs p (averaged over xi and r)
prob_p = np.mean(probabilities, axis=(1, 2))
axes[0].plot(p1, 1 - prob_p, 'b-', linewidth=2)
axes[0].set_xlabel('Momentum (p)', fontsize=12)
axes[0].set_ylabel('Escape Probability', fontsize=12)
axes[0].set_title('Escape Probability vs p', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Plot vs xi (averaged over p and r)
prob_xi = np.mean(probabilities, axis=(0, 2))
axes[1].plot(xi1, 1 - prob_xi, 'r-', linewidth=2)
axes[1].set_xlabel(r'Pitch Angle Cosine ($\xi$)', fontsize=12)
axes[1].set_ylabel('Escape Probability', fontsize=12)
axes[1].set_title(r'Escape Probability vs $\xi$', fontsize=12)
axes[1].grid(True, alpha=0.3)

# Plot vs r (averaged over p and xi)
prob_r = np.mean(probabilities, axis=(0, 1))
axes[2].plot(r1, 1 - prob_r, 'g-', linewidth=2)
axes[2].set_xlabel('Radial Position (r)', fontsize=12)
axes[2].set_ylabel('Escape Probability', fontsize=12)
axes[2].set_title('Escape Probability vs r', fontsize=12)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/probability_1d_profiles.png', dpi=300, bbox_inches='tight')
plt.close()

print("Evaluation completed!")
print("Generated plots:")
print("- figures/probability_p_vs_r.png")
print("- figures/probability_xi_vs_r.png") 
print("- figures/probability_1d_profiles.png")

# Print some statistics
print(f"\nStatistics:")
print(f"Overall escape probability: {(1 - probabilities.mean()):.3f}")
print(f"Escape probability range: [{(1 - probabilities.max()):.3f}, {(1 - probabilities.min()):.3f}]")
print(f"High escape regions (>0.8): {np.mean(1 - probabilities > 0.8):.3f} of parameter space")
print(f"Low escape regions (<0.2): {np.mean(1 - probabilities < 0.2):.3f} of parameter space")