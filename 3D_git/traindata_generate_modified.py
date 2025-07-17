import torch
import numpy as np
import os
from scipy.stats import norm
from tqdm import tqdm  # For progress bars
from scipy.io import loadmat


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




   


def RE_3dfun(T, dt,  dim, batch_size,  train):
    """
    Generates training or testing data for 3D plasma simulation
    
    Parameters:
    T (float): Terminal time
    dt (float): Time step for sampling
    Ns (int): Number of samples
    dim (int): Dimension (3 for p,xi,r)
    initial (ndarray): Initial conditions for testing
    train (bool): Whether generating training or testing data
    
    Returns:
    For training: x_data, y_data
    For testing: None, None, trajectory
    """
    # 3D test: dr = b dt + sigma dWt. particle pusher (MC method)
    # Domain parameters
    r_max = 1       # max of radius
    p_min = 0.5
    p_max = 5
    xi_min = -1
    xi_max = 1
    
    pr_min = [p_min, xi_min, 0]   
    pr_max = [p_max, xi_max, r_max]
    
    # SDE parameters
    mc_dt = 0.005      # dt
    t_needed = int(dt / mc_dt)
    nt = int(T / mc_dt)
    
    # For storing output data
    if train:
        all_x_data = []
        all_y_data = []
        # Y0_data = np.random.uniform(low=pr_min, high=pr_max, size=(total_samples, dim))

        base_path = 'data/3Dmaxwell_initial'
        all_initial = []  # List to store tensors

        for i in range(1, 11):
            file_name = os.path.join(base_path, f"RE3DInitial_maxwell_T{i}.mat")
            mat_data = loadmat(file_name)  # Load each .mat file
            Y0_data = mat_data['Y0']
            all_initial.append(Y0_data)  # Append the tensor to the list

        AllY0_data = np.concatenate(all_initial, axis=0)
       
    else:
        # For testing, we'll collect all sampled trajectories
        sampled_trajectories = []

        # Use specified initial condition for testing
        base_path = 'data/3Dmaxwell_initial'
        file_name = os.path.join(base_path, f"RE3DInitial_maxwell_T{10}.mat")
        mat_data = loadmat(file_name)  # Load each .mat file
        AllY0_data = mat_data['Y0']
    
    total_samples = AllY0_data.shape[0]  ## redefine test data size according to the size of RE2DInitial_maxwell_T{10}.mat
    
    num_batches = int(np.ceil(total_samples / batch_size))
    print(f"Processing {total_samples} samples in {num_batches} batches of size {batch_size}")

    # Process each batch
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        # Calculate actual batch size (might be smaller for last batch)
        start_index = batch_idx * batch_size
        end_index = min(total_samples, (batch_idx + 1) * batch_size)
        current_batch_size = end_index - start_index
        
        
        x_ini = AllY0_data[start_index:end_index]
        
        # Initialize trajectories for this batch
        x_end = np.copy(x_ini)
        flag = np.ones(current_batch_size)
        n_flag = current_batch_size
        
        # For storing trajectory points at sampled times only (memory-efficient)
        # We'll store every t_needed steps, plus the initial point
        num_sampled_steps = nt // t_needed + 1
        sampled_trajectory = np.zeros((num_sampled_steps, current_batch_size, dim))
        sampled_trajectory[0] = x_end  # Store initial state
        
        # For tracking which step to record
        sample_idx = 1
        
        # Run the SDE simulation
        for i in tqdm(range(1, nt+1), desc=f"Simulating batch {batch_idx+1}/{num_batches}", leave=False):
            if n_flag <= 0:
                break
                
            # Get indices of valid trajectories
            idx_flag = np.where(flag == 1)[0]
            x_temp = x_end[idx_flag]
            
            # Compute drift and diffusion
            drift_dt = drift_3d(x_temp)
            diff_dt = diffusion_3d(x_temp)
            
            # Generate random noise
            Wt = np.random.normal(0, np.sqrt(mc_dt), (int(n_flag), dim))
            
            # Update state
            x_end[idx_flag] += drift_dt * mc_dt + diff_dt * Wt
            
            # Apply boundary conditions
            row_out = np.where(x_end[:, 2] >= r_max)[0]
            x_end[:, 2] = np.abs(x_end[:, 2])
            x_end[x_end[:, 0] < p_min, 0] = p_min
            x_end[x_end[:, 1] >= 1, 1] -= 2 * (x_end[x_end[:, 1] >= 1, 1] - 1)
            x_end[x_end[:, 1] <= -1, 1] += 2 * (-1 - x_end[x_end[:, 1] <= -1, 1])
            
            # Mark trajectories that exceed max radius as invalid
            flag[row_out] = 0
            n_flag = np.sum(flag)
            
            # Store the state at sampled time points
            if i % t_needed == 0:
                sampled_trajectory[sample_idx] = x_end
                sample_idx += 1
        
        # Filter out invalid trajectories
        valid_indices = np.where(flag == 1)[0]
        if len(valid_indices) == 0:
            print(f"Warning: Batch {batch_idx+1} has no valid trajectories")
            continue
        
        valid_trajectory_temp = sampled_trajectory[:sample_idx, valid_indices, :] ## shape: (num_sampled_steps, current_batch_size, dim)
        
        valid_trajectory_temp_par = valid_trajectory_temp[:,:,0]*valid_trajectory_temp[:,:,1]
        valid_trajectory_temp_per = valid_trajectory_temp[:,:,0]*np.sqrt(1-valid_trajectory_temp[:,:,1]**2)

        # First, reshape both array to add a new dimension
        valid_trajectory_par = np.expand_dims(valid_trajectory_temp_par, axis=-1)  # Now shape: (num_sampled_steps, current_batch_size, 1)
        valid_trajectory_per = np.expand_dims(valid_trajectory_temp_per, axis=-1)  # Now shape: (num_sampled_steps, current_batch_size, 1)

        r_component = np.expand_dims(valid_trajectory_temp[:,:,2], axis=-1)

        # Concatenate along the last dimension
        valid_trajectory = np.concatenate([valid_trajectory_par, valid_trajectory_per, r_component], axis=-1)    # Result: (num_sampled_steps, current_batch_size, 3)




        if train:
            # Create input-output pairs
            x_batch_data = []
            y_batch_data = []
            
            for i in range(valid_trajectory.shape[0] - 1):
                x_batch_data.append(valid_trajectory[i])
                y_batch_data.append(valid_trajectory[i + 1])
            
            if x_batch_data:  # Check if we have any data
                x_batch_data = np.vstack(x_batch_data)
                y_batch_data = np.vstack(y_batch_data)
                
                # Remove any invalid data points
                valid_indices = np.where(~np.isnan(x_batch_data).any(axis=1) & ~np.isnan(y_batch_data).any(axis=1))[0]
                if len(valid_indices) > 0:
                    all_x_data.append(x_batch_data[valid_indices])
                    all_y_data.append(y_batch_data[valid_indices])
        else:
            # For testing, collect the trajectory
            sampled_trajectories.append(valid_trajectory)
    
    if train:
        # Combine all batches
        if all_x_data:
            x_data = np.vstack(all_x_data)
            y_data = np.vstack(all_y_data)
            print(f"Generated {len(x_data)} training pairs")
            return x_data, y_data, None
        else:
            print("No valid training data was generated")
            return np.array([]), np.array([]), None
    else:
        # For testing, combine all trajectories along the batch dimension
        if sampled_trajectories:
            # Find the trajectory with the most time steps
            max_steps = max(traj.shape[0] for traj in sampled_trajectories)
            
            # Combine while padding shorter trajectories
            combined_trajectories = []
            for traj in sampled_trajectories:
                if traj.shape[0] < max_steps:
                    # Pad with NaN values
                    padded = np.full((max_steps, traj.shape[1], dim), np.nan)
                    padded[:traj.shape[0]] = traj
                    combined_trajectories.append(padded)
                else:
                    combined_trajectories.append(traj)
            
            # Concatenate along the batch dimension
            result = np.concatenate(combined_trajectories, axis=1)
            print(f"Generated testing trajectories with shape: {result.shape}")
            return None, None, result
        else:
            print("No valid testing trajectories were generated")
            return None, None, np.array([])





# Main section to replace the existing code in 3d_SDE_plasmas_learning.py

#=========================================================================
#    3D Plasma SDE parameters
#=========================================================================
x_dim = 3  # Dimensions: (p, xi, r)
p_min = 0.5
p_max = 5.0

sde_T = 20.0   # Total simulation time
sde_dt = 0.2   # Time step for sampling

batch_size = 80000

# Format the datadir and figdir paths
datadir = os.path.join( f'data_{int(sde_dt*100):02d}')
figdir = os.path.join(f'fig_{int(sde_dt*100):02d}')

# Create the directories
make_folder(datadir)
make_folder(figdir)

# -------------Generate training data-------------
print("Generating training data...")
x_sample, y_sample, _ = RE_3dfun(sde_T, sde_dt,  x_dim, batch_size,  train=True)
print(f"Generated {len(x_sample)} input-output pairs")

# Save training data
np.save(os.path.join(datadir, 'x_sample.npy'), x_sample)
np.save(os.path.join(datadir, 'y_sample.npy'), y_sample)
np.save(os.path.join(datadir, 'sde_dt.npy'), np.array([sde_dt]))  # Save as array

print("Training data saved successfully")

# -------------Generate testing data-------------

print("\nGenerating testing data...")
_, _, trajectory = RE_3dfun(sde_T, sde_dt, x_dim, batch_size,  train=False)
print(f"Generated testing trajectory with shape: {trajectory.shape}")

# Save testing trajectory
np.save(os.path.join(datadir, 'test_trajectory.npy'), trajectory)
print("Testing trajectory saved successfully")

# -------------Plot initial and final distributions-------------
import matplotlib.pyplot as plt

# Plot histograms of initial conditions
for ii in range(x_dim):
    plt.figure(figsize=(10, 6))
    plt.hist(x_sample[:, ii], bins=50, alpha=0.7)
    plt.title(f'Distribution of dimension {ii+1} in training data')
    var_names = ['p (momentum)', 'xi (pitch angle cosine)', 'r (radius)']
    plt.xlabel(var_names[ii])
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(figdir, f'histogram_dim{ii+1}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Histograms saved successfully")
    print("All data generation complete!")