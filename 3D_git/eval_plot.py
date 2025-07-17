import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.io import loadmat
from scipy.stats import norm
from tqdm import tqdm  # For progress bars
# 1. Setup

x_dim = 3
sde_dt = 0.2
datadir = os.path.join(f'data_{int(sde_dt * 100):02d}')
figdir = os.path.join(f'fig_{int(sde_dt*100):02d}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_output = 'output/'   # where to save trained models
root_data = 'data/'       # where the datasets are
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




#original one
# def RE_3dfun(T, dt, idx_maxwell, dim, batch_size,  train):
#     """
#     Generates training or testing data for 3D plasma simulation
    
#     Parameters:
#     T (float): Terminal time
#     dt (float): Time step for sampling
#     Ns (int): Number of samples
#     dim (int): Dimension (3 for p,xi,r)
#     initial (ndarray): Initial conditions for testing
#     train (bool): Whether generating training or testing data
    
#     Returns:
#     For training: x_data, y_data
#     For testing: None, None, trajectory
#     """
#     # 3D test: dr = b dt + sigma dWt. particle pusher (MC method)
#     # Domain parameters
#     r_max = 1       # max of radius
#     p_min = 0.5

    
#     # SDE parameters
#     mc_dt = 0.005      # dt
#     t_needed = int(dt / mc_dt)
#     nt = int(T / mc_dt)
    
#     # For storing output data
#     if train:
#         all_x_data = []
#         all_y_data = []
#         # Y0_data = np.random.uniform(low=pr_min, high=pr_max, size=(total_samples, dim))

#         base_path = 'data/3Dmaxwell_initial'
#         all_initial = []  # List to store tensors

#         for i in range(1, 11):
#             file_name = os.path.join(base_path, f"RE3DInitial_maxwell_T{i}.mat")
#             mat_data = loadmat(file_name)  # Load each .mat file
#             Y0_data = mat_data['Y0']
#             all_initial.append(Y0_data)  # Append the tensor to the list
        
#         AllY0_data = np.concatenate(all_initial, axis=0)
       
#     else:
#         # For testing, we'll collect all sampled trajectories
#         sampled_trajectories = []

#         # Use specified initial condition for testing
#         base_path = 'data/3Dmaxwell_initial'
#         file_name = os.path.join(base_path, f"RE3DInitial_maxwell_T{idx_maxwell}.mat")
#         mat_data = loadmat(file_name)  # Load each .mat file
#         AllY0_data = mat_data['Y0']
#         X0_transform = np.zeros_like(AllY0_data)
#         X0_transform[:,0] =AllY0_data[:,0]*AllY0_data[:,1]
#         X0_transform[:,1] =AllY0_data[:,0]*np.sqrt(1-AllY0_data[:,1]**2)
#         X0_transform[:,2] =AllY0_data[:,2]

#     total_samples = AllY0_data.shape[0]  ## redefine test data size according to the size of RE2DInitial_maxwell_T{10}.mat
#     num_batches = int(np.ceil(total_samples / batch_size))
#     print(f"Processing {total_samples} samples in {num_batches} batches of size {batch_size}")

#     # Process each batch
#     for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
#         # Calculate actual batch size (might be smaller for last batch)
#         start_index = batch_idx * batch_size
#         end_index = min(total_samples, (batch_idx + 1) * batch_size)
#         current_batch_size = end_index - start_index
#         x_ini = AllY0_data[start_index:end_index]
        
#         # Initialize trajectories for this batch
#         x_end = np.copy(x_ini)
#         flag = np.ones(current_batch_size)
#         n_flag = current_batch_size
        
#         # For storing trajectory points at sampled times only (memory-efficient)
#         # We'll store every t_needed steps, plus the initial point
#         num_sampled_steps = nt // t_needed + 1
#         sampled_trajectory = np.zeros((num_sampled_steps, current_batch_size, dim))
#         sampled_trajectory[0] = x_end  # Store initial state
        
#         # For tracking which step to record
#         sample_idx = 1
        
#         # Run the SDE simulation
#         for i in tqdm(range(1, nt+1), desc=f"Simulating batch {batch_idx+1}/{num_batches}", leave=False):
#             if n_flag <= 0:
#                 break
                
#             # Get indices of valid trajectories
#             idx_flag = np.where(flag == 1)[0]
#             x_temp = x_end[idx_flag]
            
#             # Compute drift and diffusion
#             drift_dt = drift_3d(x_temp)
#             diff_dt = diffusion_3d(x_temp)
            
#             # Generate random noise
#             Wt = np.random.normal(0, np.sqrt(mc_dt), (int(n_flag), dim))
            
#             # Update state
#             x_end[idx_flag] += drift_dt * mc_dt + diff_dt * Wt
            
#             # Apply boundary conditions
#             row_out = np.where(x_end[:, 2] >= r_max)[0]
#             x_end[:, 2] = np.abs(x_end[:, 2])
#             x_end[x_end[:, 0] < p_min, 0] = p_min
#             x_end[x_end[:, 1] >= 1, 1] -= 2 * (x_end[x_end[:, 1] >= 1, 1] - 1)
#             x_end[x_end[:, 1] <= -1, 1] += 2 * (-1 - x_end[x_end[:, 1] <= -1, 1])
            
#             # Mark trajectories that exceed max radius as invalid
#             flag[row_out] = 0
#             n_flag = np.sum(flag)
            
#             # Store the state at sampled time points
#             if i % t_needed == 0:
#                 sampled_trajectory[sample_idx] = x_end
#                 sample_idx += 1
        
#         # Filter out invalid trajectories
#         valid_indices = np.where(flag == 1)[0]
#         if len(valid_indices) == 0:
#             print(f"Warning: Batch {batch_idx+1} has no valid trajectories")
#             continue
        
#         valid_trajectory_temp = sampled_trajectory[:sample_idx, valid_indices, :] ## shape: (num_sampled_steps, current_batch_size, dim)
        
#         valid_trajectory_temp_par = valid_trajectory_temp[:,:,0]*valid_trajectory_temp[:,:,1]
#         valid_trajectory_temp_per = valid_trajectory_temp[:,:,0]*np.sqrt(1-valid_trajectory_temp[:,:,1]**2)

#         # First, reshape both array to add a new dimension
#         valid_trajectory_par = np.expand_dims(valid_trajectory_temp_par, axis=-1)  # Now shape: (num_sampled_steps, current_batch_size, 1)
#         valid_trajectory_per = np.expand_dims(valid_trajectory_temp_per, axis=-1)  # Now shape: (num_sampled_steps, current_batch_size, 1)

#         r_component = np.expand_dims(valid_trajectory_temp[:,:,2], axis=-1)

#         # Concatenate along the last dimension
#         valid_trajectory = np.concatenate([valid_trajectory_par, valid_trajectory_per, r_component], axis=-1)    # Result: (num_sampled_steps, current_batch_size, 3)


#         if train:
#             # Create input-output pairs
#             x_batch_data = []
#             y_batch_data = []
            
#             for i in range(valid_trajectory.shape[0] - 1):
#                 x_batch_data.append(valid_trajectory[i])
#                 y_batch_data.append(valid_trajectory[i + 1])
            
#             if x_batch_data:  # Check if we have any data
#                 x_batch_data = np.vstack(x_batch_data)
#                 y_batch_data = np.vstack(y_batch_data)
                
#                 # Remove any invalid data points
#                 valid_indices = np.where(~np.isnan(x_batch_data).any(axis=1) & ~np.isnan(y_batch_data).any(axis=1))[0]
#                 if len(valid_indices) > 0:
#                     all_x_data.append(x_batch_data[valid_indices])
#                     all_y_data.append(y_batch_data[valid_indices])
#         else:
#             # For testing, collect the trajectory
#             sampled_trajectories.append(valid_trajectory)
    
#     if train:
#         # Combine all batches
#         if all_x_data:
#             x_data = np.vstack(all_x_data)
#             y_data = np.vstack(all_y_data)
#             print(f"Generated {len(x_data)} training pairs")
#             return x_data, y_data, None
#         else:
#             print("No valid training data was generated")
#             return np.array([]), np.array([]), None
#     else:
#         # For testing, combine all trajectories along the batch dimension
#         if sampled_trajectories:
#             # Find the trajectory with the most time steps
#             max_steps = max(traj.shape[0] for traj in sampled_trajectories)
            
#             # Combine while padding shorter trajectories
#             combined_trajectories = []
#             for traj in sampled_trajectories:
#                 if traj.shape[0] < max_steps:
#                     # Pad with NaN values
#                     padded = np.full((max_steps, traj.shape[1], dim), np.nan)
#                     padded[:traj.shape[0]] = traj
#                     combined_trajectories.append(padded)
#                 else:
#                     combined_trajectories.append(traj)
            
#             # Concatenate along the batch dimension
#             result = np.concatenate(combined_trajectories, axis=1)
#             print(f"Generated testing trajectories with shape: {result.shape}")
#             return AllY0_data, X0_transform, result
#         else:
#             print("No valid testing trajectories were generated")
            
#             return None, None, np.array([])
        





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

# 4. Load trained model
FN = FN_Net(x_dim*2, x_dim, 50).to(device)
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
    savedir = root_output
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
        self.dim = 4
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


        
def plot_diffusion_eval_3d(
    prediction, y_true_transform, idx_maxwell, sde_T, figdir,
    Npath_true, Npath_pred, n_vals=10
):
    os.makedirs(figdir, exist_ok=True)

    # --- Transform to (theta, p, r) ---
    y_true = np.zeros((Npath_true, 3))
    y_true[:, 0] = np.arctan2(y_true_transform[:, 1], y_true_transform[:, 0])  # theta
    y_true[:, 1] = np.sqrt(y_true_transform[:, 0]**2 + y_true_transform[:, 1]**2)  # p
    y_true[:, 2] = y_true_transform[:, 2]  # r

    y_pred = np.zeros((Npath_pred, 3))
    y_pred[:, 0] = np.arctan2(prediction[:, 1], prediction[:, 0])  # theta
    y_pred[:, 1] = np.sqrt(prediction[:, 0]**2 + prediction[:, 1]**2)  # p
    y_pred[:, 2] = prediction[:, 2]  # r

    p_min = 0.5
    y_pred[:,2] = np.abs(y_pred[:,2])
    y_pred[y_pred[:,2] > 1,2] = 1
    y_pred[y_pred[:,1] < p_min, 1] = p_min


    # --- Plot 1D Marginal PDFs for (theta, p, r) ---
    fig, axes = plt.subplots(1, 3, figsize=(3 * 6, 5))
    labels = [r'$\theta$', 'p', 'r']
    for i in range(3):
        ax = axes[i]
        kde_true = gaussian_kde(y_true[:, i])
        kde_pred = gaussian_kde(y_pred[:, i])
        delta = (np.max(y_true[:, i]) - np.min(y_true[:, i])) / 100
        x_vals = np.linspace(np.min(y_true[:, i]) - delta, np.max(y_true[:, i]) + delta, 1000)
        ax.plot(x_vals, kde_true(x_vals), label='Reference', color='blue')
        ax.plot(x_vals, kde_pred(x_vals), label='Learned', color='black')
        ax.set_xlabel(labels[i], fontsize=20)
        # Set axis limits
        if i == 0:
            ax.set_xlim(0, np.pi)
        elif i == 1:
            ax.set_xlim(0.5, 10)
        elif i == 2:
            ax.set_xlim(0, 1)
        ax.tick_params(labelsize=12)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, f'marginal_PDF_3d_maxwell_{idx_maxwell:.2f}_T{sde_T}.png'), dpi=300)
    plt.close()

    # --- Plot 2D Contour plots for pairs: (theta, p), (p, r), (r, theta) ---
    pair_indices = [(0, 1), (1, 2), (2, 0)]
    pair_labels = [(r'$\theta$', 'p'), ('p', 'r'), ('r', r'$\theta$')]
    pair_names = ['theta_p', 'p_r', 'r_theta']

    for (i, j), (xlabel, ylabel), name in zip(pair_indices, pair_labels, pair_names):
        fig, axs = plt.subplots(1, 2, figsize=(2 * 6, 6))
        min_len = min(len(y_true), len(y_pred), 70000)
        idx_sample = np.random.choice(min_len, min_len, replace=False)

        title_options = [
        [f'2d log10 pdf by MC at t={sde_T}', f'2d pdf by MC at t={sde_T}'],
        [f'2d log10 pdf by Diffusion at t={sde_T}', f'2d pdf by Diffusion at t={sde_T}']
]
        for k, data in enumerate([y_true, y_pred]):
            kde = gaussian_kde(data[idx_sample][:, [i, j]].T)
            deltaX = (max(data[:, i]) - min(data[:, i]))/50
            deltaY = (max(data[:, j]) - min(data[:, j]))/50
            xi, yi = np.mgrid[
                np.min(data[:, i])-deltaX:np.max(data[:, i])+deltaX:50j,
                np.min(data[:, j])-deltaY:np.max(data[:, j])+deltaY:50j
            ]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
            if name == 'r_theta':
                custom_levels = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
                axs[k].contourf(xi, yi, zi.reshape(xi.shape), levels=custom_levels, cmap='jet')
                cset = axs[k].contour(xi, yi, zi.reshape(xi.shape), levels=custom_levels, colors='black', linewidths=0.5)
                title = title_options[k][1]
                axs[k].set_title(title, fontsize=16)
            else:
                custom_levels = np.array([-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5])
                axs[k].contourf(xi, yi, np.log10(zi + 1e-18).reshape(xi.shape), levels=custom_levels, cmap='jet')
                cset = axs[k].contour(xi, yi, np.log10(zi + 1e-18).reshape(xi.shape), levels=custom_levels, colors='black', linewidths=0.5)
                title = title_options[k][0]
                axs[k].set_title(title, fontsize=16)

            axs[k].clabel(cset, inline=1, fontsize=12)
            axs[k].set_xlabel(xlabel, fontsize=20)
            axs[k].set_ylabel(ylabel, fontsize=20)
            axs[k].tick_params(labelsize=20)
            # Set axis limits for contour plots
            if name == 'theta_p':
                axs[k].set_xlim(0, np.pi)
                axs[k].set_ylim(0.5, 10)
                axs[k].set_xticks([0,np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
                axs[k].set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',r'$\pi$'])
                axs[k].set_xlabel(r'pitch angle $\theta$', fontsize=20)
                axs[k].set_ylabel('momentum p', fontsize=20)
            elif name == 'p_r':
                axs[k].set_xlim(0.5, 10)
                axs[k].set_ylim(0, 1)
                axs[k].set_ylabel(r'minor radius $r$', fontsize=20)
                axs[k].set_xlabel('momentum p', fontsize=20)
            elif name == 'r_theta':
                axs[k].set_xlim(0, 1)
                axs[k].set_ylim(0, np.pi)
                axs[k].set_yticks([0,np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
                axs[k].set_yticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',r'$\pi$'])
                axs[k].set_ylabel(r'pitch angle $\theta$', fontsize=20)
                axs[k].set_xlabel(r'minor radius $r$', fontsize=20)

        plt.tight_layout()
        plt.savefig(os.path.join(figdir, f'3D_{name}_maxwell_{idx_maxwell:.2f}_T{sde_T}.png'), dpi=300)
        plt.close()


#----------------------------------------------------------------------------------------------------
#  Main
#-----------------------------------------------------------------------------------------------------
# Set parameters
# Define path to saved file



# 2. Load normalization info & diff_scale
data_inf = torch.load(os.path.join(datadir, 'data_inf.pt'), weights_only=False)
xTrain_mean = data_inf['xTrain_mean'].to(device)
xTrain_std = data_inf['xTrain_std'].to(device)
yTrain_mean = data_inf['yTrain_mean'].to(device)
yTrain_std = data_inf['yTrain_std'].to(device)
diff_scale = data_inf['diff_scale']
print(diff_scale)


idx_maxwell = 10
ode_path_file = os.path.join(datadir, f'ode_path_max{idx_maxwell}.npy')


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
    np.save(ode_path_file, ode_path_true)
    save_data = {
    'true_init':true_init,
    'true_init_transform': true_init_transform,
    'ode_path_true': ode_path_true
}
    np.save(ode_path_file, save_data)
    print(f"Saved generated RE path to {ode_path_file}")


print("ode_path_true shape is: ", ode_path_true.shape)

y_true_transform = ode_path_true[-1,:,:]
y_true_transform = y_true_transform[y_true_transform[:,2] < 1,:]
print('y_true shape:', y_true_transform.shape)

x_pred_new = torch.tensor(true_init_transform, dtype=torch.float32, device=device)

Npath_pred0=true_init_transform.shape[0]

for jj in range(ode_time_steps):
    prediction = FN( (torch.hstack((x_pred_new,torch.randn(Npath_pred0,x_dim).to(device,dtype= torch.float32)))-xTrain_mean)/xTrain_std  ) * yTrain_std + yTrain_mean 
    prediction = ( prediction.to('cpu').detach().numpy() /diff_scale +x_pred_new.to('cpu').detach().numpy()  )
    x_pred_new= torch.tensor( prediction  ).to(device,dtype= torch.float32)

with open(root_data + 'weight_mean_std.npy', 'rb') as f:
    mu_weight = np.load(f)
    s_weight = np.load(f)

Escape = load_weight_model()
t_column = np.full((true_init.shape[0], 1), sde_T)
true_init_with_time = np.concatenate([true_init, t_column], axis=1)
test0 = (true_init_with_time-mu_weight)/s_weight
test0 = torch.tensor(test0, dtype=torch.float32).to(device)
weight_pred = Escape(test0).to('cpu').detach().numpy().flatten()
random_values = np.random.uniform(0, 1, size=len(weight_pred))
mask_to_keep = weight_pred > random_values.flatten()
prediction = prediction[mask_to_keep,:]
Npath_true = y_true_transform.shape[0]
Npath_pred=prediction.shape[0]
print(Npath_true)
print(Npath_pred)

# plot for T=30 (prediction)
plot_diffusion_eval_3d(
    prediction, y_true_transform, idx_maxwell, sde_T, figdir,
     Npath_true, Npath_pred, n_vals=10
)





# plot for early 
t_plot = 16
ode_time_steps_plot = int(np.floor(t_plot/sde_dt))
y_true_transform = ode_path_true[ode_time_steps_plot,:,:]
y_true_transform = y_true_transform[y_true_transform[:,2] < 1,:]

x_pred_new = torch.tensor(true_init_transform, dtype=torch.float32, device=device)
Npath_pred0=true_init_transform.shape[0]

for jj in range(ode_time_steps_plot):
    prediction = FN( (torch.hstack((x_pred_new,torch.randn(Npath_pred0,x_dim).to(device,dtype= torch.float32)))-xTrain_mean)/xTrain_std  ) * yTrain_std + yTrain_mean 
    prediction = ( prediction.to('cpu').detach().numpy() /diff_scale +x_pred_new.to('cpu').detach().numpy()  ) 
    x_pred_new= torch.tensor( prediction  ).to(device,dtype= torch.float32)

t_column = np.full((true_init.shape[0], 1), t_plot)
true_init_with_time = np.concatenate([true_init, t_column], axis=1)
test0 = (true_init_with_time-mu_weight)/s_weight
test0 = torch.tensor(test0, dtype=torch.float32).to(device)
weight_pred = Escape(test0).to('cpu').detach().numpy().flatten()
random_values = np.random.uniform(0, 1, size=len(weight_pred))
mask_to_keep = weight_pred > random_values.flatten()
prediction = prediction[mask_to_keep,:]
Npath_true = y_true_transform.shape[0]
Npath_pred=prediction.shape[0]
print(Npath_true)
print(Npath_pred)

plot_diffusion_eval_3d(
    prediction, y_true_transform, idx_maxwell, t_plot, figdir,
     Npath_true, Npath_pred, n_vals=10
)




# plot for early 
t_plot = 8
ode_time_steps_plot = int(np.floor(t_plot/sde_dt))
y_true_transform = ode_path_true[ode_time_steps_plot,:,:]
y_true_transform = y_true_transform[y_true_transform[:,2] < 1,:]

x_pred_new = torch.tensor(true_init_transform, dtype=torch.float32, device=device)
Npath_pred0=true_init_transform.shape[0]

for jj in range(ode_time_steps_plot):
    prediction = FN( (torch.hstack((x_pred_new,torch.randn(Npath_pred0,x_dim).to(device,dtype= torch.float32)))-xTrain_mean)/xTrain_std  ) * yTrain_std + yTrain_mean 
    prediction = ( prediction.to('cpu').detach().numpy() /diff_scale +x_pred_new.to('cpu').detach().numpy()  ) 
    x_pred_new= torch.tensor( prediction  ).to(device,dtype= torch.float32)

t_column = np.full((true_init.shape[0], 1), t_plot)
true_init_with_time = np.concatenate([true_init, t_column], axis=1)
test0 = (true_init_with_time-mu_weight)/s_weight
test0 = torch.tensor(test0, dtype=torch.float32).to(device)
weight_pred = Escape(test0).to('cpu').detach().numpy().flatten()
random_values = np.random.uniform(0, 1, size=len(weight_pred))
mask_to_keep = weight_pred > random_values.flatten()
prediction = prediction[mask_to_keep,:]
Npath_true = y_true_transform.shape[0]
Npath_pred=prediction.shape[0]
print(Npath_true)
print(Npath_pred)

plot_diffusion_eval_3d(
    prediction, y_true_transform, idx_maxwell, t_plot, figdir,
     Npath_true, Npath_pred, n_vals=10
)
