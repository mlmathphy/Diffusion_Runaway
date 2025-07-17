import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# 1. Setup


x_dim = 1
sde_dt = 0.1
figdir = os.path.join(f'fig_{int(sde_dt*100):02d}/')
root_data = os.path.join(f'data_{int(sde_dt * 100):02d}/')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def load_weight_checkpoint(filepath):
    """
    Load checkpoint
    """
    checkpoint = torch.load(filepath, weights_only=True)
    model = EscapeModel().to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def load_weight_model():
    """
    Load model
    """
    savedir = root_data
    filename = 'NN_time_weight'

    print(savedir + filename + '.pt')
    device = torch.device("cuda")
    model = load_weight_checkpoint(savedir + filename + '.pt')
    model.to(device)
    model.eval()
    return model


# Define deeper model with dropout
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

with open(root_data + 'weight_mean_std.npy', 'rb') as f:
    mu_weight = np.load(f)
    s_weight = np.load(f)

Escape = load_weight_model()



x_min, x_max = 0.01, 6-0.01  # Define the range for p

num_points = 101  # Number of points per axis

# Generate the mesh grid for x1 and x2
P = np.linspace(x_min, x_max, num_points)

# Flatten and stack into a 2D array
true_init_with_time = np.stack([P.ravel()], axis=1)

test0 = (true_init_with_time-mu_weight)/s_weight
test0 = torch.tensor(test0, dtype=torch.float32).to(device)
probabilities = Escape(test0).to('cpu').detach().numpy().reshape(P.shape)
print(P)
print(1-probabilities)



# Plot 1: Escape probability of (p, r)

fig, ax = plt.subplots()
ax.plot(P, 1 - probabilities)

ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('prob', fontsize=16)
plt.savefig(os.path.join(figdir, 'probability_x.png'), dpi=300, bbox_inches='tight')
plt.close()
