# Modified RE_3dfun function to generate datasets like the 1D example

import torch
import numpy as np
import os
from scipy.stats import norm
from tqdm import tqdm
from scipy.io import loadmat

datadir = os.path.join(f'data')

def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def drift_3d(x_end):
    drift = np.zeros_like(x_end)
    p, xi, r = x_end[:, 0], x_end[:, 1], x_end[:, 2]

    Z, E0, tau, T_tilde, mc2, Tf_hat, D0, rm, LD = 1.0, 1 / 2000, 6000.0, 3.0, 500.0, 0.05, 0.01, 0.5, 0.1

    delta_tilde = np.sqrt(2 * T_tilde / mc2)
    delta = np.sqrt(2 * Tf_hat / mc2)
    vT_bar = np.sqrt(Tf_hat / T_tilde)

    lnA_hat = 14.9 - 0.5 * np.log(0.28) + np.log(Tf_hat)
    lnA_tilde = 14.9 - 0.5 * np.log(0.28) + np.log(T_tilde)
    vee_bar = (T_tilde / Tf_hat)**1.5 * lnA_hat / lnA_tilde
    E = E0 * (T_tilde / Tf_hat)**1.5

    gamma = np.sqrt(1 + (delta_tilde * p)**2)
    xx = 1 / vT_bar * p / gamma

    phi = norm.cdf(xx, 0, np.sqrt(0.5)) - norm.cdf(-xx, 0, np.sqrt(0.5))
    psi = (phi - xx * (2 / np.sqrt(np.pi)) * np.exp(-xx**2)) / (2 * xx**2)

    ca = np.maximum(vee_bar * vT_bar**2 * psi / xx, 1e-10)
    cb = np.maximum(0.5 * vee_bar * vT_bar**2 * (Z + phi - psi + delta**4 * xx**2 * 0.5) / xx, 1e-10)
    cf = 2 * vee_bar * vT_bar * psi

    dp = 0.001
    p1 = p + dp
    gamma_p1 = np.sqrt(1 + (delta_tilde * p1)**2)
    xx_p1 = 1 / vT_bar * p1 / gamma_p1
    phi_p1 = norm.cdf(xx_p1, 0, np.sqrt(0.5)) - norm.cdf(-xx_p1, 0, np.sqrt(0.5))
    psi_p1 = (phi_p1 - xx_p1 * (2 / np.sqrt(np.pi)) * np.exp(-xx_p1**2)) / (2 * xx_p1**2)
    CA_p1 = np.maximum(vee_bar * vT_bar**2 * psi_p1 / xx_p1, 1e-10)

    drift[:, 0] = E * xi - gamma * p / tau * (1 - xi**2) - cf + 2 / p * ca + (CA_p1 - ca) / dp
    drift[:, 1] = E * (1 - xi**2) / p + xi * (1 - xi**2) / tau / gamma - 2 * xi * cb / p**2

    dr = 1e-4
    delta_p = 2
    F_r = 0.5 * (1 + np.tanh((r - rm) / LD))
    F_r_dr = 0.5 * (1 + np.tanh((r - dr - rm) / LD))
    drift[:, 2] = D0 * (F_r - F_r_dr) / dr * np.exp(-(p / delta_p)**2)
    return drift

def diffusion_3d(x_end):
    diff = np.zeros_like(x_end)
    p, xi, r = x_end[:, 0], x_end[:, 1], x_end[:, 2]

    Z, T_tilde, mc2, Tf_hat = 1.0, 3, 500.0, 0.05
    delta_tilde = np.sqrt(2 * T_tilde / mc2)
    delta = np.sqrt(2 * Tf_hat / mc2)
    vT_bar = np.sqrt(Tf_hat / T_tilde)

    lnA_hat = 14.9 - 0.5 * np.log(0.28) + np.log(Tf_hat)
    lnA_tilde = 14.9 - 0.5 * np.log(0.28) + np.log(T_tilde)
    vee_bar = (T_tilde / Tf_hat)**1.5 * lnA_hat / lnA_tilde

    gamma = np.sqrt(1 + (delta_tilde * p)**2)
    xx = 1 / vT_bar * p / gamma

    phi = norm.cdf(xx, 0, np.sqrt(0.5)) - norm.cdf(-xx, 0, np.sqrt(0.5))
    psi = (phi - xx * (2 / np.sqrt(np.pi)) * np.exp(-xx**2)) / (2 * xx**2)

    ca = np.maximum(vee_bar * vT_bar**2 * psi / xx, 1e-10)
    cb = np.maximum(0.5 * vee_bar * vT_bar**2 * (Z + phi - psi + delta**4 * xx**2 * 0.5) / xx, 1e-10)

    diff[:, 0] = np.sqrt(2 * ca)
    diff[:, 1] = np.sqrt(2 * cb) * np.sqrt(1 - xi**2) / p
    D0, rm, LD = 0.01, 0.5, 0.1
    delta_p = 2
    F_r = 0.5 * (1 + np.tanh((r - rm) / LD))
    diff[:, 2] = np.sqrt(2 * D0 * F_r * np.exp(-(p / delta_p)**2))
    return diff

def RE_3dfun(T, dt, dim, batch_size, train):
    # Domain parameters
    r_max, p_min, p_max, xi_min, xi_max = 1, 0.5, 5, -1, 1
    pr_min, pr_max = [p_min, xi_min, 0], [p_max, xi_max, r_max]

    # SDE parameters
    mc_dt = 0.005
    t_needed = int(dt / mc_dt)
    nt = int(T / mc_dt)
    N_snap = int(np.floor(T / dt) + 1)
    print(f"Number of snapshots: {N_snap}")

    # Load initial conditions
    if train:
        base_path = 'data/3Dmaxwell_initial'
        all_initial = [loadmat(os.path.join(base_path, f"RE3DInitial_maxwell_T{i}.mat"))['Y0'] for i in range(2,11,2)]
        AllY0_data = np.concatenate(all_initial, axis=0)
    else:
        mat_data = loadmat(os.path.join('data/3Dmaxwell_initial', f"RE3DInitial_maxwell_T10.mat"))
        AllY0_data = mat_data['Y0']

    total_samples = AllY0_data.shape[0]
    num_batches = int(np.ceil(total_samples / batch_size))
    print(f"Processing {total_samples} samples in {num_batches} batches")

    # Initialize storage for 1D-style datasets
    flow_x = []         # record X_n    in pair active
    flow_y = []         # record X_n+1  in pair active
    exit_x = []         # record X_n in pair
    exit_delta = []     # record X_n exit status

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start, end = batch_idx * batch_size, min(total_samples, (batch_idx + 1) * batch_size)
        x_ini = AllY0_data[start:end]
        x_end = np.copy(x_ini)

        current_batch_size = x_ini.shape[0]
        flag = np.ones(current_batch_size)
        n_flag = current_batch_size

        # Arrays to store trajectories and flags at snapshot times
        sampled_trajectory = np.zeros((current_batch_size, N_snap, dim))
        exit_trajectory = np.zeros((current_batch_size, N_snap))

        # Initialize storage arrays
        sampled_trajectory[:, 0, :] = x_end
        exit_trajectory[:, 0] = flag

        sample_idx = 1

        for i in tqdm(range(1, nt + 1), desc=f"Simulating batch {batch_idx+1}", leave=False):
            if n_flag <= 0:
                break
                
            if i % 1000 == 0:
                print(f"Step {i}/{nt}, Active particles: {n_flag}")

            idx_flag = np.where(flag == 1)[0]
            if len(idx_flag) == 0:
                break
                
            x_temp = x_end[idx_flag]
            drift_dt = drift_3d(x_temp)
            diff_dt = diffusion_3d(x_temp)
            Wt = np.random.normal(0, np.sqrt(mc_dt), (int(n_flag), dim))
            x_end[idx_flag] += drift_dt * mc_dt + diff_dt * Wt

            # Apply boundary conditions
            x_end[idx_flag, 2] = np.abs(x_end[idx_flag, 2])  # Reflect r at 0
            x_end[x_end[:, 0] < p_min, 0] = p_min  # Reflect p at p_min
            x_end[x_end[:, 1] >= 1, 1] = 2 - x_end[x_end[:, 1] >= 1, 1]  # Reflect xi at 1
            x_end[x_end[:, 1] <= -1, 1] = -2 - x_end[x_end[:, 1] <= -1, 1]  # Reflect xi at -1
            
            # Check for exit condition (r >= r_max)
            row_out = np.where(x_end[:, 2] >= r_max)[0]
            flag[row_out] = 0
            n_flag = np.sum(flag)

            # Store the state at sampled time points
            if i % t_needed == 0:
                sampled_trajectory[:, sample_idx, :] = x_end
                exit_trajectory[:, sample_idx] = flag
                sample_idx += 1

        # Collect active consecutive pairs (X_n, X_{n+1}) for this batch
        for t_idx in range(min(sample_idx, N_snap) - 1):
            for particle_idx in range(current_batch_size):
                flag_x_n = exit_trajectory[particle_idx, t_idx]         # flag at time n
                flag_x_n_plus_1 = exit_trajectory[particle_idx, t_idx + 1]  # flag at time n+1
                
                # If both are active (both flags = 1)
                if flag_x_n == 1 and flag_x_n_plus_1 == 1:
                    x_n = sampled_trajectory[particle_idx, t_idx, :]       # X_n (3D vector)
                    x_n_plus_1 = sampled_trajectory[particle_idx, t_idx + 1, :]  # X_{n+1} (3D vector)
                    
                    # Add the consecutive pair
                    flow_x.append(x_n)
                    flow_y.append(x_n_plus_1)

        # Collect exit prediction pairs (X_n, flag_{n+1}) for this batch
        for t_idx in range(min(sample_idx, N_snap) - 1):
            for particle_idx in range(current_batch_size):
                flag_x_n = exit_trajectory[particle_idx, t_idx]         # flag at time n
                flag_x_n_plus_1 = exit_trajectory[particle_idx, t_idx + 1]  # flag at time n+1
                
                # If particle is active at time n
                if flag_x_n == 1:
                    x_n = sampled_trajectory[particle_idx, t_idx, :]  # Position at time n (3D vector)
                    
                    # Record the pair: position and next flag status
                    exit_x.append(x_n)
                    exit_delta.append(flag_x_n_plus_1)

    # Convert to numpy arrays
    flow_x = np.array(flow_x)
    flow_y = np.array(flow_y)
    exit_x = np.array(exit_x)
    exit_delta = np.array(exit_delta)

    print(f"Collected {len(flow_x)} active consecutive pairs")
    print(f"Collected {len(exit_x)} active particles")
    print(f"Exited particles: {np.sum(exit_delta==0)} out of {len(exit_x)} ({100*np.mean(exit_delta==0):.2f}%)")
    print(f"Remaining active: {np.sum(exit_delta==1)} out of {len(exit_x)} ({100*np.mean(exit_delta==1):.2f}%)")
    
    return flow_x, flow_y, exit_x, exit_delta

# Parameters
batch_size = 160000
dim = 3
sde_T = 20.0   # Total simulation time
sde_dt = 0.2   # Time step for sampling

# Create data directory
make_folder(datadir)

# Generate training data
print("Generating training data...")
x_sample, y_sample, exit_x, exit_delta = RE_3dfun(sde_T, sde_dt, dim, batch_size, train=True)
print(f"Generated {len(x_sample)} input-output pairs")

# Save training data in the same format as 1D example
# np.save(os.path.join(datadir, 'x_sample.npy'), x_sample)
# np.save(os.path.join(datadir, 'y_sample.npy'), y_sample)
np.save(os.path.join(datadir, 'sde_dt.npy'), np.array([sde_dt]))  # Save as array
np.save(os.path.join(datadir, 'exit_x.npy'), exit_x)
np.save(os.path.join(datadir, 'exit_delta.npy'), exit_delta)
print("Training data saved successfully")

print(f"Dataset shapes:")
print(f"x_sample: {x_sample.shape}")
print(f"y_sample: {y_sample.shape}")
print(f"exit_x: {exit_x.shape}")
print(f"exit_delta: {exit_delta.shape}")