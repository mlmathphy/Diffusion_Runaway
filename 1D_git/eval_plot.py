import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.io import loadmat
from scipy.stats import norm
from tqdm import tqdm  # For progress bars


from matplotlib.lines import Line2D
sde_dt = 0.1

# 1. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_dim = 1

datadir = os.path.join(f'data_{int(sde_dt * 100):02d}')
figdir = os.path.join(f'fig_{int(sde_dt*100):02d}')


print(datadir)
def BM_1dfun(T, dt, dim, Nsample, idx_initial):
    # 1D test: dx = dWt. particle pusher (MC method)
    # Domain parameters
    x_min = 0
    x_max = 10
        
    # SDE parameters
    mc_dt = 0.0001      # dt
    t_needed = int(dt / mc_dt)
    Nt = int(np.floor(T / mc_dt) + 1)
    N_snap = int(np.floor(T / dt) + 1)

    # For storing output data
    if idx_initial==1:
        x_ini = np.random.uniform(low=x_min, high=x_max, size=(Nsample, dim))
    elif idx_initial==2:
        x_ini = 5.0 * np.ones((Nsample, dim))  # Fixed: added parentheses
    elif idx_initial==3:
        x_ini = 1.0 * np.ones((Nsample, dim))  # Fixed: added parentheses
    
    x_end = np.copy(x_ini)
    flag = np.ones(Nsample)
    n_flag = Nsample

    # Fixed: proper array dimensions
    sampled_trajectory = np.zeros((Nsample, N_snap))  # record X_n in snapshot
    exit_trajectory = np.zeros((Nsample, N_snap))     # record exit in snapshot

    # Fixed: proper indexing for initial storage
    sampled_trajectory[:, 0] = x_end.flatten()
    exit_trajectory[:, 0] = flag

    # For tracking which step to record
    sample_idx = 1
    
    for ii in range(0, Nt):
        if n_flag <= 0:
            break

        if ii % 500 == 0:
            print(ii, Nt)

        # Update particle positions
        idx_flag = np.where(flag == 1)[0]
        Wt = np.random.normal(0, np.sqrt(mc_dt), (int(n_flag), dim))
        x_end[idx_flag] += Wt
        
        # Fixed: proper boolean indexing
        row_out = np.where((x_end >= x_max) | (x_end <= x_min))[0]
        # Mark trajectories that exceed domain as invalid
        flag[row_out] = 0
        n_flag = np.sum(flag)

        # Store the state at sampled time points
        if (ii+1) % t_needed == 0:
            sampled_trajectory[:, sample_idx] = x_end.flatten()
            exit_trajectory[:, sample_idx] = flag
            sample_idx += 1

    return x_ini, sampled_trajectory, exit_trajectory


# 3. Define your NN class again (same as training)
class FN_Net(nn.Module):
    def __init__(self, input_dim, output_dim, hid_size):
        super(FN_Net, self).__init__()
        self.input = nn.Linear(input_dim, hid_size)
        self.fc1 = nn.Linear(hid_size, hid_size)
        self.output = nn.Linear(hid_size, output_dim)

    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.fc1(x))
        return self.output(x)

# 4. Load trained model
FN = FN_Net(x_dim*2, x_dim, 64).to(device)
FN.load_state_dict(torch.load(os.path.join(datadir, 'FN.pth'), map_location=device, weights_only=True))
FN.eval()

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
    savedir = datadir
    filename = '/NN_time_weight'

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

Escape = load_weight_model()
with open(datadir + '/weight_mean_std.npy', 'rb') as f:
    mu_weight = np.load(f)
    s_weight = np.load(f)

#----------------------------------------------------------------------------------------------------
#  Main
#-----------------------------------------------------------------------------------------------------
# Set parameters
# Define path to saved file
data_inf = torch.load(os.path.join(datadir, 'data_inf.pt'), weights_only=False)
xTrain_mean = data_inf['xTrain_mean'].to(device)
xTrain_std = data_inf['xTrain_std'].to(device)
yTrain_mean = data_inf['yTrain_mean'].to(device)
yTrain_std = data_inf['yTrain_std'].to(device)
diff_scale = data_inf['diff_scale']
print(diff_scale)

Nsample = 100000


idx_initial = 2
ode_path_file = os.path.join(datadir, f'ode_path_{idx_initial}.npy')

ytrue_pred_list = []

sde_T = 3.0

ode_time_steps = int(np.floor(sde_T/sde_dt))
print('ode_time_steps is:', ode_time_steps)

# Check if file exists
if os.path.exists(ode_path_file):
    print(f"Loading existing RE path from {ode_path_file}")
    loaded = np.load(ode_path_file, allow_pickle=True).item()
    true_init = loaded['true_init']
    sampled_trajectory = loaded['sampled_trajectory']
    exit_trajectory = loaded['exit_trajectory']

else:
    print("File not found. Running RE_3dfun to generate data...")
    true_init, sampled_trajectory, exit_trajectory = BM_1dfun(sde_T, sde_dt, x_dim, Nsample, idx_initial)
    save_data = {
    'true_init':true_init,
    'sampled_trajectory': sampled_trajectory,
    'exit_trajectory': exit_trajectory
}
    np.save(ode_path_file, save_data)
    print(f"Saved generated RE path to {ode_path_file}")


print("sampled_trajectory shape is: ", sampled_trajectory.shape)


print( (np.where(exit_trajectory[:,1] == 0)[0]).shape)
y_true_transform = sampled_trajectory[:,-1]
exit_true_transform = exit_trajectory[:,-1]
y_true_transform = y_true_transform[exit_true_transform==1]
# print('y_true shape:', y_true_transform.shape)

x_pred_new = torch.tensor(true_init, dtype=torch.float32, device=device)

mu_weight = torch.tensor(mu_weight, dtype=torch.float32, device=device)
s_weight = torch.tensor(s_weight, dtype=torch.float32, device=device)
print(yTrain_mean)
for jj in range(ode_time_steps):
    print(jj)
    # Fix 1: Keep operations on GPU or move to CPU consistently
    test0 = (x_pred_new - mu_weight) / s_weight
    weight_pred = Escape(test0)

    # Fix 2: Use torch.rand instead of torch.random.uniform
    random_values = torch.rand(len(weight_pred), device=device)  # Keep on same device
    mask_to_keep = weight_pred.flatten() > random_values.flatten()
    false_indices = torch.where(mask_to_keep == False)[0]

    # print(false_indices.size())
    # print( x_pred_new[false_indices])
    # print(weight_pred[false_indices])
    Npath_pred0 = x_pred_new.size(0)
    prediction = FN((torch.hstack((x_pred_new, torch.randn(Npath_pred0, x_dim).to(device, dtype=torch.float32))) - xTrain_mean) / xTrain_std) * yTrain_std + yTrain_mean
    # exit()


    # Fix 3: Handle tensor operations consistently
    prediction = prediction / diff_scale + x_pred_new
    # print(torch.min(prediction))
    # print(torch.max(prediction))

    # Apply mask while still on GPU
    prediction = prediction[mask_to_keep, :]
    
    # Update for next iteration (keep on GPU)
    x_pred_new = prediction.clone()
    
    # Optional: Print progress
    if jj % 5 == 0:
        print(f"Step {jj}, remaining particles: {x_pred_new.size(0)}")

Npath_true = y_true_transform.shape
Npath_pred = prediction.flatten().shape


print(Npath_true)
print(Npath_pred)
prediction_cpu = prediction.cpu().detach().numpy().flatten()
# Plot KDE comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# KDE for true data
if len(y_true_transform) > 1:
    kde_true = gaussian_kde(y_true_transform)
    x_range = np.linspace(min(np.min(y_true_transform), np.min(prediction_cpu)), 
                         max(np.max(y_true_transform), np.max(prediction_cpu)), 1000)
    kde_true_vals = kde_true(x_range)
    ax.plot(x_range, kde_true_vals, label='True KDE', color='blue', linewidth=2)

# KDE for predicted data
if len(prediction_cpu) > 1:
    kde_pred = gaussian_kde(prediction_cpu)
    kde_pred_vals = kde_pred(x_range)
    ax.plot(x_range, kde_pred_vals, label='Predicted KDE', color='red', linewidth=2)


ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_title('KDE Comparison: True vs Predicted')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path_file = os.path.join(figdir,f'kde_comparison_{idx_initial}.png')
plt.savefig(fig_path_file, dpi=300, bbox_inches='tight')
plt.show()
