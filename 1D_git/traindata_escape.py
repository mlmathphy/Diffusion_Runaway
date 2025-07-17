import torch
import numpy as np
import os
from scipy.stats import norm
from tqdm import tqdm  # For progress bars
from scipy.io import loadmat

x_min = 0.0
x_max = 6.0
x_dim = 1  # Dimensions: x
sde_T = 0.1   # Total simulation time
sde_dt = 0.1   # Time step for sampling

def make_folder(folder):
    """
    Creates given folder (or path) if it doesn't exist.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device',device)

root_data = 'data/'

seed = 12345
np.random.seed(seed)


def make_folder(folder):
    """
    Creates given folder (or path) if it doesn't exist.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


def create_weighted_samples(x_min, x_max, n_points, total_samples):
    # Create linspace points
    points = np.linspace(x_min, x_max, n_points)
    print(points)
    # Calculate distance from center
    center = (x_min + x_max) / 2
    distances_from_center = np.abs(points - center)
    
    # Convert to weights (closer to boundaries = higher weight)
    max_distance = np.max(distances_from_center)
    weights = distances_from_center / max_distance  # 0 to 1
    weights = weights + 0.1  # Add small base weight so center isn't completely ignored
    
    # Normalize weights to sum to total_samples
    weights = weights / np.sum(weights) * total_samples
    n_repeats = np.round(weights).astype(int)
    
    # Adjust to match exact total if needed
    diff = total_samples - np.sum(n_repeats)
    if diff > 0:
        n_repeats[-diff:] += 1
    elif diff < 0:
        n_repeats[:abs(diff)] -= 1
    
    # Create samples
    samples = np.repeat(points, n_repeats)
    return samples.reshape(-1, 1)
   

def BM_1dfun(T, dt, dim, Nsample, train):
    # 1D test: dx = dWt. particle pusher (MC method)
    # Domain parameters
        
    # SDE parameters
    # mc_dt = 0.0005      # dt
    mc_dt = 0.0001
    t_needed = int(dt / mc_dt)
    Nt = int(np.floor(T / mc_dt) + 1)
    N_snap = int(np.floor(T / dt) + 1)
    print(N_snap)
    # For storing output data
    if train:
        # # x_ini = np.random.uniform(low=x_min, high=x_max, size=(Nsample, dim))
        # n_points = 401
        # n_repeats = 4000
        # points = np.linspace(x_min, x_max, n_points)
        # samples = np.repeat(points, n_repeats)
        # x_ini = samples.reshape(-1, 1)
        x_ini = create_weighted_samples(x_min, x_max, 201, 2000000)

    else:
        x_ini = 5.0 * np.ones((Nsample, dim))  # Fixed: added parentheses
    


    Nsample = 2000000
    x_end = np.copy(x_ini)
    flag = np.ones(Nsample)
    n_flag = Nsample

    # Fixed: proper array dimensions
    sampled_trajectory = np.zeros((Nsample, N_snap))  # record X_n in snapshot
    exit_trajectory = np.zeros((Nsample, N_snap))     # record exit in snapshot

    # Initialize storage arrays
    flow_x = []         # record X_n    in pair active
    flow_y = []         # record X_n+1  in pair active
    exit_x = []         # record X_n in pair
    exit_delta = []     # record X_n exit status

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
            print(ii)
            sampled_trajectory[:, sample_idx] = x_end.flatten()
            exit_trajectory[:, sample_idx] = flag
            sample_idx += 1

    # Collect active consecutive pairs (X_n, X_{n+1})
    for t_idx in range(N_snap - 1):
        for particle_idx in range(Nsample):
            flag_x_n = exit_trajectory[particle_idx, t_idx]         # flag at time n
            flag_x_n_plus_1 = exit_trajectory[particle_idx, t_idx + 1]  # flag at time n+1
            
            # If both are active (both flags = 1)
            if flag_x_n == 1 and flag_x_n_plus_1 == 1:
                x_n = sampled_trajectory[particle_idx, t_idx]       # X_n
                x_n_plus_1 = sampled_trajectory[particle_idx, t_idx + 1]  # X_{n+1}
                
                # Add the consecutive pair
                flow_x.append(x_n)
                flow_y.append(x_n_plus_1)

    # Collect exit prediction pairs (X_n, flag_{n+1})
    for t_idx in range(N_snap - 1):
        for particle_idx in range(Nsample):
            flag_x_n = exit_trajectory[particle_idx, t_idx]         # flag at time n
            flag_x_n_plus_1 = exit_trajectory[particle_idx, t_idx + 1]  # flag at time n+1
            
            # If particle is active at time n
            if flag_x_n == 1:
                x_n = sampled_trajectory[particle_idx, t_idx]  # Position at time n
                
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






#=========================================================================
#    3D Plasma SDE parameters
#=========================================================================

Nsample = 2000000

# Format the datadir and figdir paths
datadir = os.path.join( f'data_{int(sde_dt*100):02d}')
figdir = os.path.join(f'fig_{int(sde_dt*100):02d}')

# Create the directories
make_folder(datadir)
make_folder(figdir)

# -------------Generate training data-------------
print("Generating training data...")
x_sample, y_sample, exit_x, exit_delta  = BM_1dfun(sde_T, sde_dt, x_dim, Nsample, train=True)
print(f"Generated {len(x_sample)} input-output pairs")


indices = np.where(exit_x == 0.06)[0]
print(indices)
print(sum(exit_delta[indices])/indices.size)
indices = np.where(exit_x == 0.12)[0]
print(indices)
print(sum(exit_delta[indices])/indices.size)
# Save training data
np.save(os.path.join(datadir, 'exit_x.npy'), exit_x)
np.save(os.path.join(datadir, 'exit_delta.npy'), exit_delta)
print("Training data saved successfully")

