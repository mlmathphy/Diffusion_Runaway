import torch
import numpy as np
import os
from scipy.stats import norm
from tqdm import tqdm  # For progress bars
from scipy.io import loadmat

x_min = 0.0
x_max = 6.0
x_dim = 1  # Dimensions: x
sde_T = 3.0   # Total simulation time
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


   

def BM_1dfun(T, dt, dim, Nsample, train):
    # 1D test: dx = dWt. particle pusher (MC method)
    # Domain parameters
        
    # SDE parameters
    mc_dt = 0.0005      # dt
    t_needed = int(dt / mc_dt)
    Nt = int(np.floor(T / mc_dt) + 1)
    N_snap = int(np.floor(T / dt) + 1)
    print(N_snap)
    # For storing output data
    if train:
        x_ini = np.random.uniform(low=x_min, high=x_max, size=(Nsample, dim))
    else:
        x_ini = 5.0 * np.ones((Nsample, dim))  # Fixed: added parentheses
    
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

Nsample = 100000

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

# Save training data
np.save(os.path.join(datadir, 'x_sample.npy'), x_sample)
np.save(os.path.join(datadir, 'y_sample.npy'), y_sample)
np.save(os.path.join(datadir, 'sde_dt.npy'), np.array([sde_dt]))  # Save as array
np.save(os.path.join(datadir, 'exit_x.npy'), exit_x)
np.save(os.path.join(datadir, 'exit_delta.npy'), exit_delta)
print("Training data saved successfully")



# # -------------Generate testing data-------------

# print("\nGenerating testing data...")
# _, _, trajectory = RE_3dfun(sde_T, sde_dt, x_dim, batch_size,  train=False)
# print(f"Generated testing trajectory with shape: {trajectory.shape}")

# # Save testing trajectory
# np.save(os.path.join(datadir, 'test_trajectory.npy'), trajectory)
# print("Testing trajectory saved successfully")

