import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.io import savemat

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

# Update for 2D data directory
root_data = 'data_2d/'  # Adjust based on your 2D data folder

# Set random seed
np.random.seed(12345)
torch.manual_seed(12345)

# --- Load 2D data ---
dim = 2  # Changed from 1 to 2 for 2D problem

# Load 2D exit data
with open(root_data + 'exit_x.npy', 'rb') as f:
    XX = np.load(f)  # Shape: (N, 2) - contains (x1, x2) positions
with open(root_data + 'exit_delta.npy', 'rb') as f:
    YY = np.load(f)  # Shape: (N,) - exit flags

print(f"Original data shapes: X={XX.shape}, Y={YY.shape}")
print(f"Original exit statistics: {np.sum(YY==0)} exited, {np.sum(YY==1)} still active")

# Filter data where X[:,1] is in range [0, 0.7]
mask = (XX[:, 1] >= 0.0) & (XX[:, 1] <= 0.8)
# mask = (XX[:, 1] >= 1.2) & (XX[:, 1] <= 2.0)
XX_filtered = XX[mask]
YY_filtered = YY[mask]

print(f"Filtered data shapes: X={XX_filtered.shape}, Y={YY_filtered.shape}")
print(f"Filtered exit statistics: {np.sum(YY_filtered==0)} exited, {np.sum(YY_filtered==1)} still active")
print(f"Retained {np.sum(mask)}/{len(mask)} ({100*np.sum(mask)/len(mask):.1f}%) of original data")

# Create final subset
X = XX_filtered  # Shape: (N_filtered, 2) for 2D positions
Y = YY_filtered.reshape(-1, 1)  # Shape: (N_filtered, 1) for labels


# Normalize features for both dimensions (using filtered data)
mu_weight = X.mean(axis=0)  # Shape: (2,) - mean for each dimension
s_weight = X.std(axis=0)    # Shape: (2,) - std for each dimension

# Save normalization parameters
with open(root_data + 'weight_mean_std_2d_lower.npy', 'wb') as f:
    np.save(f, mu_weight)
    np.save(f, s_weight)

print(f"Normalization - Mean: {mu_weight}, Std: {s_weight}")
print(f"X[:,1] range in filtered data: [{X[:,1].min():.3f}, {X[:,1].max():.3f}]")

# Normalize both x1 and x2 dimensions
X = (X - mu_weight) / s_weight

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

# Train/Validation split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=123)
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

# Batch size - keeping the same as 1D version
batch_size = 2048
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_model(model):
    make_folder(root_data)
    filename = 'NN_time_weight_2d_lower'  # Updated filename for 2D
    checkpoint = {'state_dict': model.state_dict()}
    torch.save(checkpoint, root_data + filename + '.pt')

# Updated model for 2D input
class EscapeModel2D(nn.Module):
    def __init__(self):
        super(EscapeModel2D, self).__init__()
        self.dim = dim  # Now 2 for 2D input
        self.hid_size = 512  # Keep same hidden size
        self.net = nn.Sequential(
            nn.Linear(self.dim, self.hid_size),  # Input layer: 2 -> 256
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(self.hid_size, self.hid_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(self.hid_size, self.hid_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(self.hid_size, 1),  # Output layer: 256 -> 1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = EscapeModel2D().to(device)

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# Same loss function as 1D version
criterion = nn.BCELoss()

# Same optimizer settings as 1D version
optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)

# Same scheduler settings as 1D version
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=5
)

# Training parameters - same as 1D version
num_epochs = 100
train_losses = []
val_losses = []

print("Starting 2D escape probability training...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    # Progress logging
    if epoch % 5 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")

save_model(model)

# Enhanced visualization for 2D case
plt.figure(figsize=(15, 10))

# Training curves
plt.subplot(2, 3, 1)
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(val_losses, label='Val Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Curves')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(val_losses, label='Val Loss', alpha=0.8)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.legend()
plt.title('Training Curves (Log Scale)')
plt.grid(True, alpha=0.3)

# 2D data visualization
plt.subplot(2, 3, 3)
# Denormalize for plotting
X_plot = X_val.numpy() * s_weight + mu_weight
Y_plot = Y_val.numpy().flatten()

# Plot training data colored by exit status
active_mask = Y_plot == 1
exited_mask = Y_plot == 0

if np.sum(active_mask) > 0:
    plt.scatter(X_plot[active_mask, 0], X_plot[active_mask, 1], 
               c='blue', alpha=0.6, s=1, label='Still active')
if np.sum(exited_mask) > 0:
    plt.scatter(X_plot[exited_mask, 0], X_plot[exited_mask, 1], 
               c='red', alpha=0.6, s=1, label='Exited')

plt.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Exit boundaries')
plt.axhline(y=1, color='black', linestyle='--', alpha=0.7)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Validation Data: Actual Exit Status')
plt.legend()
plt.xlim(-np.pi, np.pi)
plt.ylim(-0.1, 1.1)

# Model predictions visualization
plt.subplot(2, 3, 4)
model.eval()
with torch.no_grad():
    val_outputs = model(X_val.to(device)).cpu().numpy().flatten()

# Create prediction plot
scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=val_outputs, 
                     cmap='RdYlBu', alpha=0.6, s=1)
plt.colorbar(scatter, label='Predicted Probability (Stay Active)')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.axhline(y=1, color='black', linestyle='--', alpha=0.7)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Model Predictions')
plt.xlim(-np.pi, np.pi)
plt.ylim(-0.1, 1.1)

# Prediction accuracy by region
plt.subplot(2, 3, 5)
val_preds = (val_outputs > 0.5).astype(float)
correct = (val_preds == Y_plot)

plt.scatter(X_plot[correct, 0], X_plot[correct, 1], 
           c='green', alpha=0.6, s=1, label='Correct')
plt.scatter(X_plot[~correct, 0], X_plot[~correct, 1], 
           c='red', alpha=0.6, s=1, label='Incorrect')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.axhline(y=1, color='black', linestyle='--', alpha=0.7)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Prediction Accuracy')
plt.legend()
plt.xlim(-np.pi, np.pi)
plt.ylim(-0.1, 1.1)

# Probability distribution
plt.subplot(2, 3, 6)
plt.hist(val_outputs[Y_plot == 1], bins=50, alpha=0.7, label='Actual: Active', density=True)
plt.hist(val_outputs[Y_plot == 0], bins=50, alpha=0.7, label='Actual: Exited', density=True)
plt.xlabel('Predicted Probability (Stay Active)')
plt.ylabel('Density')
plt.title('Prediction Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('escape_2d_training_results_lower.png', dpi=150, bbox_inches='tight')
plt.show()

# Final statistics
print(f"\nFinal Train Loss: {train_losses[-1]:.6f}")
print(f"Final Val Loss: {val_losses[-1]:.6f}")
print(f"Improvement from epoch 0: Train={train_losses[0]-train_losses[-1]:.6f}, Val={val_losses[0]-val_losses[-1]:.6f}")

# Detailed accuracy analysis
model.eval()
with torch.no_grad():
    val_outputs = model(X_val.to(device))
    val_preds = (val_outputs > 0.5).float()
    accuracy = (val_preds == Y_val.to(device)).float().mean()
    
    # Calculate precision, recall for each class
    true_positive = ((val_preds == 1) & (Y_val.to(device) == 1)).float().sum()
    false_positive = ((val_preds == 1) & (Y_val.to(device) == 0)).float().sum()
    true_negative = ((val_preds == 0) & (Y_val.to(device) == 0)).float().sum()
    false_negative = ((val_preds == 0) & (Y_val.to(device) == 1)).float().sum()
    
    precision_active = true_positive / (true_positive + false_positive + 1e-8)
    recall_active = true_positive / (true_positive + false_negative + 1e-8)
    precision_exit = true_negative / (true_negative + false_negative + 1e-8)
    recall_exit = true_negative / (true_negative + false_positive + 1e-8)
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Precision (Active): {precision_active:.4f}, Recall (Active): {recall_active:.4f}")
    print(f"Precision (Exit): {precision_exit:.4f}, Recall (Exit): {recall_exit:.4f}")



# Create a simple prediction function for testing
def predict_escape_probability(x1, x2, model_path=None, normalize_params_path=None):
    """
    Predict escape probability for given 2D coordinates
    
    Args:
        x1, x2: coordinates (can be arrays)
        normalize_params_path: path to normalization parameters (optional)
    
    Returns:
        Predicted probability of staying active (not exiting)
    """
    if model_path is None:
        model_path = root_data + 'NN_time_weight_2d.pt'
    if normalize_params_path is None:
        normalize_params_path = root_data + 'weight_mean_std_2d.npy'
    
    # Load model
    model_test = EscapeModel2D()
    checkpoint = torch.load(model_path, map_location=device)
    model_test.load_state_dict(checkpoint['state_dict'])
    model_test.eval()
    
    # Load normalization parameters
    with open(normalize_params_path, 'rb') as f:
        mu = np.load(f)
        sigma = np.load(f)
    
    # Prepare input
    if np.isscalar(x1):
        x1, x2 = np.array([x1]), np.array([x2])
    
    X_test = np.column_stack([x1.flatten(), x2.flatten()])
    X_test_norm = (X_test - mu) / sigma
    X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        predictions = model_test(X_test_tensor).numpy().flatten()
    
    return predictions.reshape(x1.shape) if x1.ndim > 0 else predictions[0]

