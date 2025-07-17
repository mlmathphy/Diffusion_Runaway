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

# 1. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_output = 'output/'   # where to save trained models
root_data = 'data/'       # where the datasets are

x_dim = 3
sde_dt = 0.2
figdir = os.path.join(f'fig_{int(sde_dt*100):02d}')

def generate_combined_legend(sde_T_list, colors):
    custom_lines = []
    custom_labels = []
    for j, t_val in enumerate(sde_T_list):
        custom_lines.append(Line2D([0], [0], linestyle='--', color=colors[j], lw=2))  # True
        custom_labels.append(f'MC t={t_val}')
        custom_lines.append(Line2D([0], [0], linestyle='-', color=colors[j], lw=2))   # Pred
        custom_labels.append(f'Diffusion t={t_val}')
    return custom_lines, custom_labels

def drift_3d(x_end):
    drift = np.zeros_like(x_end) # (p,xi,r)
    p = x_end[:,0]
    xi = x_end[:,1]
    r = x_end[:,2]

    Z = 1.0
    E0 = 1/2000
    tau = 6000.0
    T_tilde = 3.0
    mc2 = 500.0
    Tf_hat = 0.05
    D0 = 0.01
    rm = 0.5
    LD = 0.1

    delta_tilde = np.sqrt(2 * T_tilde / mc2)
    delta = np.sqrt(2 * Tf_hat / mc2)

    vT_bar = np.sqrt(Tf_hat / T_tilde)

    lnA_hat = 14.9 - 1/2 * np.log(0.28) + np.log(Tf_hat)
    lnA_tilde = 14.9 - 1/2 * np.log(0.28) + np.log(T_tilde)
    vee_bar = (T_tilde / Tf_hat)**(3/2) * lnA_hat / lnA_tilde
    E = E0 * (T_tilde / Tf_hat)**(3/2)

    gamma = np.sqrt( 1 + (delta_tilde * p)**2 )
    xx = 1 / vT_bar * p / gamma

    phi = norm.cdf(xx, 0, np.sqrt(0.5)) - norm.cdf(-xx, 0, np.sqrt(0.5))
    psi = 1 / (2 * xx**2) * ( phi - xx * (2 / np.sqrt(np.pi)) * np.exp( -xx**2 ) )

    ca = vee_bar * vT_bar**2 * psi / xx
    cb = 0.5 * vee_bar * vT_bar**2 * ( Z + phi - psi + delta**4 * xx**2 * 0.5 ) / xx
    cf = 2 * vee_bar * vT_bar * psi
    ca = np.maximum(ca, 1e-10)
    cb = np.maximum(cb, 1e-10)

    dp = 0.001
    p1 = p + dp
    gamma_p1 = np.sqrt( 1 + (delta_tilde * p1)**2 )
    xx_p1 = 1 / vT_bar * p1 / gamma_p1

    phi_p1 = norm.cdf(xx_p1, 0, np.sqrt(0.5)) - norm.cdf(-xx_p1, 0, np.sqrt(0.5))
    psi_p1 = 1 / (2 * xx_p1**2) * ( phi_p1 - xx_p1 * (2 / np.sqrt(np.pi)) * np.exp( -xx_p1**2 ) )

    CA_p1 = vee_bar * vT_bar**2 * psi_p1 / xx_p1
    CA_p1 = np.maximum(CA_p1, 1e-10)

    drift[:,0] = E * xi - gamma * p / tau * (1-xi**2) - cf + 2/p * ca + (CA_p1 - ca)/dp
    drift[:,1] = E * (1-xi**2) / p + xi * (1-xi**2)/tau/gamma - 2*xi * cb / p**2
    
    dr = 1e-04
    delta_p = 2
    F_r = np.zeros_like(r)
    F_r_dr = np.zeros_like(r)
    F_r = 0.5*(1+np.tanh((r - rm)/LD))
    F_r_dr = 0.5*(1+np.tanh((r-dr - rm)/LD))
    drift[:,2] = D0*(F_r-F_r_dr)/dr*np.exp(-(p/delta_p)**2)
    return drift

def diffusion_3d(x_end):
    diff = np.zeros_like(x_end) # (p,xi,r)
    p = x_end[:,0]
    xi = x_end[:,1]
    r = x_end[:,2]

    Z = 1.0
    T_tilde = 3
    mc2 = 500.0
    Tf_hat = 0.05

    delta_tilde = np.sqrt(2 * T_tilde / mc2)
    delta = np.sqrt(2 * Tf_hat / mc2)

    vT_bar = np.sqrt(Tf_hat / T_tilde)

    lnA_hat = 14.9 - 1/2 * np.log(0.28) + np.log(Tf_hat)
    lnA_tilde = 14.9 - 1/2 * np.log(0.28) + np.log(T_tilde)
    vee_bar = (T_tilde / Tf_hat)**(3/2) * lnA_hat / lnA_tilde

    gamma = np.sqrt( 1 + (delta_tilde * p)**2 )
    xx = 1 / vT_bar * p / gamma

    phi = norm.cdf(xx, 0, np.sqrt(0.5)) - norm.cdf(-xx, 0, np.sqrt(0.5))
    psi = 1 / (2 * xx**2) * ( phi - xx * (2 / np.sqrt(np.pi)) * np.exp( -xx**2 ) )

    ca = vee_bar * vT_bar**2 * psi / xx
    cb = 0.5 * vee_bar * vT_bar**2 * ( Z + phi - psi + delta**4 * xx**2 * 0.5 ) / xx
    ca = np.maximum(ca, 1e-10)
    cb = np.maximum(cb, 1e-10)

    diff[:,0] = np.sqrt(2*ca)
    diff[:,1] = np.sqrt(2*cb) * np.sqrt( 1-xi**2 ) / p
    D0 = 0.01
    rm = 0.5
    LD = 0.1

    delta_p = 2
    F_r = np.zeros_like(r)
    F_r = 0.5*(1+np.tanh((r - rm)/LD))
    diff[:,2] = np.sqrt(2*D0*F_r*np.exp(-(p/delta_p)**2))
    return diff

def RE_3dfun(T, dt, idx_maxwell, dim, batch_size, train):
    r_max = 1
    p_min = 0.5
    mc_dt = 0.005
    t_needed = int(dt / mc_dt)
    nt = int(T / mc_dt)

    if train:
        all_x_data = []
        all_y_data = []
        base_path = 'data/3Dmaxwell_initial'
        all_initial = []
        for i in range(1, 11):
            file_name = os.path.join(base_path, f"RE3DInitial_maxwell_T{i}.mat")
            mat_data = loadmat(file_name)
            Y0_data = mat_data['Y0']
            all_initial.append(Y0_data)
        AllY0_data = np.concatenate(all_initial, axis=0)
    else:
        sampled_trajectories = []
        base_path = 'data/3Dmaxwell_initial'
        file_name = os.path.join(base_path, f"RE3DInitial_maxwell_T{idx_maxwell}.mat")
        mat_data = loadmat(file_name)
        AllY0_data = mat_data['Y0']
        X0_transform = np.zeros_like(AllY0_data)
        X0_transform[:, 0] = AllY0_data[:, 0] * AllY0_data[:, 1]
        X0_transform[:, 1] = AllY0_data[:, 0] * np.sqrt(1 - AllY0_data[:, 1]**2)
        X0_transform[:, 2] = AllY0_data[:, 2]

    total_samples = AllY0_data.shape[0]
    num_batches = int(np.ceil(total_samples / batch_size))
    print(f"Processing {total_samples} samples in {num_batches} batches of size {batch_size}")

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_index = batch_idx * batch_size
        end_index = min(total_samples, (batch_idx + 1) * batch_size)
        current_batch_size = end_index - start_index
        x_ini = AllY0_data[start_index:end_index]

        x_end = np.copy(x_ini)
        frozen_mask = np.zeros(current_batch_size, dtype=bool)

        num_sampled_steps = nt // t_needed + 1
        sampled_trajectory = np.zeros((num_sampled_steps, current_batch_size, dim))
        sampled_trajectory[0] = x_end
        sample_idx = 1

        for i in tqdm(range(1, nt + 1), desc=f"Simulating batch {batch_idx + 1}/{num_batches}", leave=False):
            idx_to_update = np.where(~frozen_mask)[0]
            if len(idx_to_update) == 0:
                break

            x_temp = x_end[idx_to_update]
            drift_dt = drift_3d(x_temp)
            diff_dt = diffusion_3d(x_temp)
            Wt = np.random.normal(0, np.sqrt(mc_dt), (len(idx_to_update), dim))
            x_end[idx_to_update] += drift_dt * mc_dt + diff_dt * Wt

            x_end[:, 2] = np.abs(x_end[:, 2])
            x_end[x_end[:, 0] < p_min, 0] = p_min
            x_end[x_end[:, 1] >= 1, 1] -= 2 * (x_end[x_end[:, 1] >= 1, 1] - 1)
            x_end[x_end[:, 1] <= -1, 1] += 2 * (-1 - x_end[x_end[:, 1] <= -1, 1])

            new_frozen = (x_end[:, 2] >= r_max) & (~frozen_mask)
            frozen_mask[new_frozen] = True

            if i % t_needed == 0:
                sampled_trajectory[sample_idx] = x_end
                sample_idx += 1

        valid_indices = np.arange(current_batch_size)
        valid_trajectory_temp = sampled_trajectory[:sample_idx, valid_indices, :]

        valid_trajectory_temp_par = valid_trajectory_temp[:, :, 0] * valid_trajectory_temp[:, :, 1]
        valid_trajectory_temp_per = valid_trajectory_temp[:, :, 0] * np.sqrt(1 - valid_trajectory_temp[:, :, 1]**2)

        valid_trajectory_par = np.expand_dims(valid_trajectory_temp_par, axis=-1)
        valid_trajectory_per = np.expand_dims(valid_trajectory_temp_per, axis=-1)
        r_component = np.expand_dims(valid_trajectory_temp[:, :, 2], axis=-1)

        valid_trajectory = np.concatenate([valid_trajectory_par, valid_trajectory_per, r_component], axis=-1)

        if train:
            x_batch_data = []
            y_batch_data = []
            for i in range(valid_trajectory.shape[0] - 1):
                x_batch_data.append(valid_trajectory[i])
                y_batch_data.append(valid_trajectory[i + 1])
            if x_batch_data:
                x_batch_data = np.vstack(x_batch_data)
                y_batch_data = np.vstack(y_batch_data)
                all_x_data.append(x_batch_data)
                all_y_data.append(y_batch_data)
        else:
            sampled_trajectories.append(valid_trajectory)

    if train:
        if all_x_data:
            x_data = np.vstack(all_x_data)
            y_data = np.vstack(all_y_data)
            print(f"Generated {len(x_data)} training pairs")
            return x_data, y_data, None
        else:
            print("No training data was generated")
            return np.array([]), np.array([]), None
    else:
        if sampled_trajectories:
            max_steps = max(traj.shape[0] for traj in sampled_trajectories)
            combined_trajectories = []
            for traj in sampled_trajectories:
                if traj.shape[0] < max_steps:
                    padded = np.full((max_steps, traj.shape[1], dim), np.nan)
                    padded[:traj.shape[0]] = traj
                    combined_trajectories.append(padded)
                else:
                    combined_trajectories.append(traj)
            result = np.concatenate(combined_trajectories, axis=1)
            print(f"Generated testing trajectories with shape: {result.shape}")
            return AllY0_data, X0_transform, result
        else:
            print("No testing trajectories were generated")
            return None, None, np.array([])

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

# 4. Load trained FN model
FN = FN_Net(x_dim*2, x_dim, 128).to(device)
FN.load_state_dict(torch.load(os.path.join(root_data, 'FN.pth'), map_location=device, weights_only=True))
FN.eval()

# Define the 3D escape model (matching our new training code)
class EscapeModel(nn.Module):
    # def __init__(self):
    #     super(EscapeModel, self).__init__()
    #     self.dim = 3  # 3D: (p, xi, r) - NO TIME COMPONENT
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
    #         nn.Sigmoid()  # Keep sigmoid for evaluation
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

def load_escape_checkpoint(filepath):
    """
    Load checkpoint for 3D escape model
    """
    try:
        checkpoint = torch.load(filepath, weights_only=False, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
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
        try:
            with open(os.path.join(root_data, 'weight_mean_std.npy'), 'rb') as f:
                mu_weight = np.load(f)
                s_weight = np.load(f)
        except FileNotFoundError:
            print("Warning: No normalization file found. Using default values.")

    
    return model, mu_weight, s_weight

def load_escape_model():
    """
    Load the trained 3D escape model
    """
    savedir = root_output
    filename = 'NN_3D_escape_model'  # Updated filename from training code
    
    filepath = os.path.join(savedir, filename + '.pt')
    print(f"Loading escape model from: {filepath}")
    
    model, mu_weight, s_weight = load_escape_checkpoint(filepath)
    model.to(device)
    model.eval()
    return model, mu_weight, s_weight

def plot_combined_kde_3d(ytrue_pred_list, figdir, idx_maxwell, n_vals=10):
    os.makedirs(figdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = [r'$\theta$', 'p', 'r']
    colors = ['blue', 'green', 'red']
    sde_T_list = [item[2] for item in ytrue_pred_list]

    for i in range(3):  # for θ, p, r
        ax = axes[i]
        for j, (y_true_raw, y_pred_raw, t_val) in enumerate(ytrue_pred_list):
            y_true = np.zeros_like(y_true_raw)
            y_true[:, 0] = np.arctan2(y_true_raw[:, 1], y_true_raw[:, 0])
            y_true[:, 1] = np.sqrt(y_true_raw[:, 0]**2 + y_true_raw[:, 1]**2)
            y_true[:, 2] = y_true_raw[:, 2]

            y_pred = np.zeros_like(y_pred_raw)
            y_pred[:, 0] = np.arctan2(y_pred_raw[:, 1], y_pred_raw[:, 0])
            y_pred[:, 1] = np.sqrt(y_pred_raw[:, 0]**2 + y_pred_raw[:, 1]**2)
            y_pred[:, 2] = y_pred_raw[:, 2]
            y_pred[:, 2] = np.clip(np.abs(y_pred[:, 2]), 0, 1)
            y_pred[y_pred[:, 1] < 0.5, 1] = 0.5

            kde_true = gaussian_kde(y_true[:, i])
            kde_pred = gaussian_kde(y_pred[:, i])
            delta = (np.max(y_true[:, i]) - np.min(y_true[:, i])) / 100
            x_vals = np.linspace(np.min(y_true[:, i]) - delta, np.max(y_true[:, i]) + delta, 1000)
            if i == 0:
                ax.plot(x_vals, kde_true(x_vals), linestyle='--', color=colors[j], linewidth=1.5)
                ax.plot(x_vals, kde_pred(x_vals), linestyle='-', color=colors[j], linewidth=1.5)
                ax.set_xticks([0,np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
                ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
                ax.set_title(r'Marginal pdf of $\theta$', fontsize=16)
                ax.set_xlabel(r'pitch angle $\theta$', fontsize=16)
            elif i == 1:
                ax.plot(x_vals, np.log10(kde_true(x_vals)+1e-8), linestyle='--', color=colors[j], linewidth=1.5)
                ax.plot(x_vals, np.log10(kde_pred(x_vals)+1e-8), linestyle='-', color=colors[j], linewidth=1.5)
                ax.set_title('Marginal log10(pdf) of p', fontsize=16)
                ax.set_xlabel('momentum p', fontsize=16)
            else:
                ax.plot(x_vals, kde_true(x_vals), linestyle='--', color=colors[j], linewidth=1.5)
                ax.plot(x_vals, kde_pred(x_vals), linestyle='-', color=colors[j], linewidth=1.5)
                ax.set_title(r'Marginal pdf of $r$', fontsize=16)
                ax.set_xlabel(r'minor radius r', fontsize=16)

        ax.set_xlabel(labels[i], fontsize=16)
        if i == 0:
            ax.set_xlim(0, np.pi)
        elif i == 1:
            ax.set_xlim(0.5, 10)
        elif i == 2:
            ax.set_xlim(0, 1)
        ax.tick_params(labelsize=12)

    # Add legend
    legend_lines, legend_labels = generate_combined_legend(sde_T_list, colors)
    axes[0].legend(legend_lines, legend_labels, fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(figdir, f'combined_kde_3d_maxwell_{idx_maxwell}.png'), dpi=300)
    plt.close()

#----------------------------------------------------------------------------------------------------
#  Main
#-----------------------------------------------------------------------------------------------------

# 2. Load normalization info & diff_scale
data_inf = torch.load(os.path.join(root_data, 'data_inf.pt'), weights_only=False)
xTrain_mean = data_inf['xTrain_mean'].to(device)
xTrain_std = data_inf['xTrain_std'].to(device)
yTrain_mean = data_inf['yTrain_mean'].to(device)
yTrain_std = data_inf['yTrain_std'].to(device)
diff_scale = data_inf['diff_scale']
# Convert diff_scale to tensor for GPU operations
diff_scale_tensor = torch.tensor(diff_scale, dtype=torch.float32, device=device)
print(f"diff_scale: {diff_scale}")

# Load the new 3D escape model
Escape, mu_weight, s_weight = load_escape_model()
print(f"Loaded escape model normalization - Mean: {mu_weight}, Std: {s_weight}")

# Convert to tensors for GPU operations
mu_weight_tensor = torch.tensor(mu_weight, dtype=torch.float32, device=device)
s_weight_tensor = torch.tensor(s_weight, dtype=torch.float32, device=device)

idx_maxwell = 10
ode_path_file = os.path.join(root_data, f'ode_path_max{idx_maxwell}.npy')

ytrue_pred_list = []
sde_T = 24.0
ode_time_steps = int(np.floor(sde_T/sde_dt))
print('ode_time_steps is:', ode_time_steps)

# Check if file exists
if os.path.exists(ode_path_file):
    print(f"Loading existing RE path from {ode_path_file}")
    loaded = np.load(ode_path_file, allow_pickle=True).item()
    true_init = loaded['true_init']
    true_init_transform = loaded['true_init_transform']
    ode_path_true = loaded['ode_path_true']
else:
    print("File not found. Running RE_3dfun to generate data...")
    true_init, true_init_transform, ode_path_true = RE_3dfun(sde_T, sde_dt, idx_maxwell, x_dim, batch_size = 80000, train=False)
    save_data = {
    'true_init':true_init,
    'true_init_transform': true_init_transform,
    'ode_path_true': ode_path_true
}
    np.save(ode_path_file, save_data)
    print(f"Saved generated RE path to {ode_path_file}")

print("ode_path_true shape is: ", ode_path_true.shape)

# Process final time step
y_true_transform = ode_path_true[-1,:,:]
y_true_transform = y_true_transform[y_true_transform[:,2] < 1,:]
print('y_true shape:', y_true_transform.shape)

x_pred_new = torch.tensor(true_init_transform, dtype=torch.float32, device=device)
# x_pred_new_ori = torch.tensor(true_init, dtype=torch.float32, device=device)
Npath_pred0 = true_init_transform.shape[0]

# Main prediction loop
for jj in range(ode_time_steps):
    print(f"Time step {jj+1}/{ode_time_steps}, particles: {x_pred_new.size(0)}")
    print(torch.max(x_pred_new[:,2]))
    print(torch.mean(x_pred_new[:,2]))
  
    # Apply escape model (3D input only, no time)
    deterministic_threshold = 0.81
    below_threshold_mask = x_pred_new[:, 2] <= deterministic_threshold

    # Initialize mask - points below threshold always stay (mask_to_keep = True)
    mask_to_keep = torch.ones(len(x_pred_new), dtype=torch.bool, device=device)

    # Only apply escape model to points above threshold
    above_threshold_mask = x_pred_new[:, 2] > deterministic_threshold
    n_above_threshold = torch.sum(above_threshold_mask)

    if n_above_threshold > 0:
        # Apply escape model only to uncertain region
        test0 = (x_pred_new[above_threshold_mask] - mu_weight_tensor) / s_weight_tensor
        weight_pred = Escape(test0)
        
        # Generate random values and apply mask for uncertain points only
        random_values = torch.rand(n_above_threshold, device=device)
        mask_uncertain = weight_pred.flatten() > random_values.flatten()
        
        # Update the full mask - uncertain points get their model prediction
        mask_to_keep[above_threshold_mask] = mask_uncertain

    print(np.sum(ode_path_true[jj,:,2]<1))
    print(sum(mask_to_keep))



    # Apply FN model prediction
    Npath_current = x_pred_new.size(0)
    prediction = FN((torch.hstack((x_pred_new, torch.randn(Npath_current, x_dim).to(device, dtype=torch.float32))) - xTrain_mean) / xTrain_std) * yTrain_std + yTrain_mean
    prediction = prediction / diff_scale_tensor + x_pred_new
    
    # Apply escape mask
    prediction = prediction[mask_to_keep, :]
    # print(prediction.size(0))
    # Apply domain constraints for 3D physics
    # For momentum p: keep >= 0.5
    # For xi: keep within [-1, 1] 
    # For r: keep >= 0 and < 1 (escape condition)

    x_pred_new = prediction.clone()
    # x_pred_new_ori = prediction.clone()
    # x_pred_new_ori[:, 1] = torch.arctan2(x_pred_new[:, 1], x_pred_new[:, 0])
    # x_pred_new_ori[:, 0] = torch.sqrt(x_pred_new[:, 0]**2 + x_pred_new[:, 1]**2)


    # x_end[:, 2] = np.abs(x_end[:, 2])
    # x_end[x_end[:, 0] < p_min, 0] = p_min
    # x_end[x_end[:, 1] >= 1, 1] -= 2 * (x_end[x_end[:, 1] >= 1, 1] - 1)
    # x_end[x_end[:, 1] <= -1, 1] += 2 * (-1 - x_end[x_end[:, 1] <= -1, 1])
    

    # keep_prediction_flag = (x_pred_new_ori[:, 0] >= 0.5) & \
    #                       (x_pred_new_ori[:, 1] >= -torch.pi) & (x_pred_new_ori[:, 1] <= torch.pi) & \
    #                       (x_pred_new_ori[:, 2] >= 0) & (x_pred_new_ori[:, 2] < 1)
    # keep_prediction_flag = (x_pred_new_ori[:, 2] < 1)
    # prediction = prediction[keep_prediction_flag, :]
    # print(prediction.size(0))

    if x_pred_new.size(0) == 0:
        print("All particles escaped or violated constraints!")
        break



# Convert final prediction to numpy
prediction = x_pred_new.cpu().detach().numpy()
Npath_true = y_true_transform.shape[0]
Npath_pred = prediction.shape[0]
print(f"Final: True particles: {Npath_true}, Predicted particles: {Npath_pred}")

ytrue_pred_list.append((y_true_transform, prediction, sde_T))

# Process earlier time steps
for t_plot in [16, 8]:
    print(f"\nProcessing t = {t_plot}")
    ode_time_steps_plot = int(np.floor(t_plot/sde_dt))
    y_true_transform = ode_path_true[ode_time_steps_plot,:,:]
    y_true_transform = y_true_transform[y_true_transform[:,2] < 1,:]

    x_pred_new = torch.tensor(true_init_transform, dtype=torch.float32, device=device)
    # x_pred_new_ori = torch.tensor(true_init, dtype=torch.float32, device=device)

    for jj in range(ode_time_steps_plot):
        # Apply escape model (3D input only)

        # Apply deterministic boundary first
        deterministic_threshold = 0.81
        below_threshold_mask = x_pred_new[:, 2] <= deterministic_threshold

        # Initialize mask - points below threshold always stay (mask_to_keep = True)
        mask_to_keep = torch.ones(len(x_pred_new), dtype=torch.bool, device=device)

        # Only apply escape model to points above threshold
        above_threshold_mask = x_pred_new[:, 2] > deterministic_threshold
        n_above_threshold = torch.sum(above_threshold_mask)

        if n_above_threshold > 0:
            # Apply escape model only to uncertain region
            test0 = (x_pred_new[above_threshold_mask] - mu_weight_tensor) / s_weight_tensor
            weight_pred = Escape(test0)
            
            # Generate random values and apply mask for uncertain points only
            random_values = torch.rand(n_above_threshold, device=device)
            mask_uncertain = weight_pred.flatten() > random_values.flatten()
            
            # Update the full mask - uncertain points get their model prediction
            mask_to_keep[above_threshold_mask] = mask_uncertain
        
        # Apply FN model prediction
        Npath_current = x_pred_new.size(0)
        prediction = FN((torch.hstack((x_pred_new, torch.randn(Npath_current, x_dim).to(device, dtype=torch.float32))) - xTrain_mean) / xTrain_std) * yTrain_std + yTrain_mean
        prediction = prediction / diff_scale_tensor + x_pred_new
        
            
        # Apply escape mask
        prediction = prediction[mask_to_keep, :]
        print(prediction.size(0))
        # Apply domain constraints for 3D physics
        # For momentum p: keep >= 0.5
        # For xi: keep within [-1, 1] 
        # For r: keep >= 0 and < 1 (escape condition)

        x_pred_new = prediction.clone()
        # x_pred_new_ori = prediction.clone()
        # x_pred_new_ori[:, 1] = torch.arctan2(x_pred_new[:, 1], x_pred_new[:, 0])
        # x_pred_new_ori[:, 0] = torch.sqrt(x_pred_new[:, 0]**2 + x_pred_new[:, 1]**2)
        # keep_prediction_flag = (x_pred_new_ori[:, 0] >= 0.5) & \
        #                     (x_pred_new_ori[:, 1] >= -torch.pi) & (x_pred_new_ori[:, 1] <= torch.pi) & \
        #                     (x_pred_new_ori[:, 2] >= 0) & (x_pred_new_ori[:, 2] < 1)
        
        # prediction = prediction[keep_prediction_flag, :]
        print(prediction.size(0))
        if x_pred_new.size(0) == 0:
            print(f"All particles escaped at step {jj+1}!")
            break
    
    # Convert to numpy for storage
    prediction = x_pred_new.cpu().detach().numpy()
    Npath_true = y_true_transform.shape[0]
    Npath_pred = prediction.shape[0]
    print(f"t={t_plot}: True particles: {Npath_true}, Predicted particles: {Npath_pred}")

    ytrue_pred_list.append((y_true_transform, prediction, t_plot))

# Reverse to chronological order
ytrue_pred_list.reverse()
plot_combined_kde_3d(ytrue_pred_list, figdir, idx_maxwell, n_vals=10)

print("Evaluation completed successfully!")
print(f"Generated plot: {os.path.join(figdir, f'combined_kde_3d_maxwell_{idx_maxwell}.png')}")