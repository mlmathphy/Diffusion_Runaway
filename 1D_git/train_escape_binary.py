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

root_data = 'data_10/'

# Set random seed
np.random.seed(12345)
torch.manual_seed(12345)

# --- Load your data ---
dim = 1  
# Load data
with open(root_data + 'exit_x.npy', 'rb') as f:
    XX = np.load(f)
with open(root_data + 'exit_delta.npy', 'rb') as f:
    YY = np.load(f)

# Create final subset
X = XX.reshape(-1, 1)
Y = YY.reshape(-1, 1)

# Save for inspection
mdic = {"loc": X, "prob": Y}
savemat("escape_prob.mat", mdic)

# Normalize features
mu_weight = X.mean(axis=0)
s_weight = X.std(axis=0)
with open(root_data + 'weight_mean_std.npy', 'wb') as f:
    np.save(f, mu_weight)
    np.save(f, s_weight)

X = (X - mu_weight) / s_weight

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

# Train/Validation split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=123)
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

# MINIMAL CHANGE 1: Slightly smaller batch size for better gradient estimates
batch_size = 2048  # Was 4096, now 2048
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_model(model):
    make_folder(root_data)
    filename = 'NN_time_weight'
    checkpoint = {'state_dict': model.state_dict()}
    torch.save(checkpoint, root_data + filename + '.pt')

# YOUR EXACT ORIGINAL MODEL - NO CHANGES
class EscapeModel(nn.Module):
    def __init__(self):
        super(EscapeModel, self).__init__()
        self.dim = dim
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

model = EscapeModel().to(device)

# YOUR EXACT ORIGINAL LOSS - NO CHANGES
criterion = nn.BCELoss()

# MINIMAL CHANGE 2: Slightly lower learning rate for better convergence
optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)  # Was 0.005, now 0.003

# MINIMAL CHANGE 3: More patient scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=5  # Was factor=0.5, patience=3
)

# MINIMAL CHANGE 4: More epochs with better monitoring
num_epochs = 50  # Was 30, now 50
train_losses = []
val_losses = []

print("Starting training...")
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

    # MINIMAL CHANGE 5: More frequent logging to see progress
    if epoch % 5 == 0:  # Was every 10, now every 5
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")

save_model(model)

# MINIMAL ADDITION: Simple training curve plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(val_losses, label='Val Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Curves')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(val_losses, label='Val Loss', alpha=0.8)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.legend()
plt.title('Training Curves (Log Scale)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFinal Train Loss: {train_losses[-1]:.6f}")
print(f"Final Val Loss: {val_losses[-1]:.6f}")
print(f"Improvement from epoch 0: Train={train_losses[0]-train_losses[-1]:.6f}, Val={val_losses[0]-val_losses[-1]:.6f}")

# Quick accuracy check
model.eval()
with torch.no_grad():
    val_outputs = model(X_val.to(device))
    val_preds = (val_outputs > 0.5).float()
    accuracy = (val_preds == Y_val.to(device)).float().mean()
    print(f"Validation Accuracy: {accuracy:.4f}")

print(f"Model saved to: {root_data}NN_time_weight.pt")