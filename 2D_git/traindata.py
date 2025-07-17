import torch
import numpy as np
import os
from scipy.stats import norm
from tqdm import tqdm  # For progress bars
import matplotlib.pyplot as plt

# Domain parameters
x_min, x_max = -np.pi, np.pi  # x1 domain
y_min, y_max = 0.0, 2.0       # x2 domain
x_dim = 2  # 2D problem
sde_T = 1   # Total simulation time (T_max from paper)
sde_dt = 0.05  # Time step for sampling

# Physical parameters from the paper
Pe = 5.      # Peclet number
epsilon = 0.0 # Parameter in velocity field
n = 2          # Parameter in velocity field

def make_folder(folder):
    """
    Creates given folder (or path) if it doesn't exist.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', device)

# Format the datadir and figdir paths
datadir = os.path.join(f'data_2d')
figdir = os.path.join(f'fig_2d')

# Create the directories
make_folder(datadir)
make_folder(figdir)
    

seed = 1234
np.random.seed(seed)

def velocity_field(x1, x2):  # Remove 't' parameter
    """
    Compute the autonomous velocity field v = (v_x1, v_x2) at position (x1, x2)
    """
    f_t = 0.0
    
    v_x1 = -np.pi * np.cos(np.pi * x2) * (np.sin(n * x1) + epsilon * f_t * np.cos(n * x1))
    v_x2 = n * np.sin(np.pi * x2) * (np.cos(n * x1) - epsilon * f_t * np.sin(n * x1))
    
    return v_x1, v_x2

def apply_periodic_bc(x1, x2):
    """
    Apply periodic boundary conditions in x1 direction
    and reflective boundary conditions in x2 direction
    """
    # Periodic in x1: [-π, π]
    x1 = ((x1 - x_min) % (x_max - x_min)) + x_min
    
    return x1, x2

def check_exit_condition(x1, x2):
    """
    Check if particles have exited the domain by reaching x2=0 or x2=1 boundaries
    Returns flag array: 1 for still inside domain, 0 for exited
    """
    flag = np.ones(len(x1))
    
    # Exit condition: particle reaches top (x2 >= 1) or bottom (x2 <= 0) boundary
    exited = (x2 >= y_max) | (x2 <= y_min)
    flag[exited] = 0
    
    return flag

def SDE_2d_advection_diffusion(T, dt, dim, Nsample, train):
    """
    2D Advection-Diffusion SDE solver
    dx1 = Pe * v_x1(x1, x2, t) * dt + dW1
    dx2 = Pe * v_x2(x1, x2, t) * dt + dW2
    """
    
    # SDE parameters
    mc_dt = 0.0005    # Fine time step for numerical integration
    t_needed = int(dt / mc_dt)
    Nt = int(np.floor(T / mc_dt) + 1)
    N_snap = int(np.floor(T / dt) + 1)
    print(f"Number of snapshots: {N_snap}")
    print(f"Total integration steps: {Nt}")
    
    # Initial conditions
    if train:
        # Random initial positions in domain
        x1_ini = np.random.uniform(low=x_min, high=x_max, size=Nsample)
        x2_ini = np.random.uniform(low=y_min, high=y_max, size=Nsample)
    else:
        # Fixed initial positions for testing
        x1_ini = np.zeros(Nsample)  # Center in x1
        x2_ini = 0.5 * np.ones(Nsample)  # Center in x2
    
    # Current positions
    x1_end = np.copy(x1_ini)
    x2_end = np.copy(x2_ini)
    flag = np.ones(Nsample)
    n_flag = Nsample
    
    # Storage arrays
    sampled_trajectory_x1 = np.zeros((Nsample, N_snap))
    sampled_trajectory_x2 = np.zeros((Nsample, N_snap))
    exit_trajectory = np.zeros((Nsample, N_snap))
    
    # Data collection arrays
    flow_x1 = []    # X1_n
    flow_x2 = []    # X2_n  
    flow_y1 = []    # X1_{n+1}
    flow_y2 = []    # X2_{n+1}
    exit_x1 = []    # X1_n for exit prediction
    exit_x2 = []    # X2_n for exit prediction
    exit_delta = [] # Exit status at n+1
    
    # Initialize storage
    sampled_trajectory_x1[:, 0] = x1_end
    sampled_trajectory_x2[:, 0] = x2_end
    exit_trajectory[:, 0] = flag
    
    sample_idx = 1
    
    print("Starting SDE integration...")
    for ii in tqdm(range(Nt)):
        if n_flag <= 0:
            print('1')
            break
            
        current_time = ii * mc_dt
        
        # Get active particles
        idx_flag = np.where(flag == 1)[0]
        n_active = len(idx_flag)
        
        if n_active == 0:
            print('2')
            break
            
        # Compute velocity field for active particles
        v_x1, v_x2 = velocity_field(x1_end[idx_flag], x2_end[idx_flag])  

        # Generate random increments
        dW1 = np.random.normal(0, np.sqrt(mc_dt), n_active)
        dW2 = np.random.normal(0, np.sqrt(mc_dt), n_active)
        
        # Update positions using SDE
        x1_end[idx_flag] += Pe * v_x1 * mc_dt + dW1
        x2_end[idx_flag] += Pe * v_x2 * mc_dt + dW2
        
        # Apply boundary conditions
        x1_end[idx_flag], x2_end[idx_flag] = apply_periodic_bc(x1_end[idx_flag], x2_end[idx_flag])
        
        # Check if particles are still valid
        flag = check_exit_condition(x1_end, x2_end)
        n_flag = np.sum(flag)
        
        # Store snapshots
        if (ii + 1) % t_needed == 0 and sample_idx < N_snap:
            sampled_trajectory_x1[:, sample_idx] = x1_end
            sampled_trajectory_x2[:, sample_idx] = x2_end
            exit_trajectory[:, sample_idx] = flag
            sample_idx += 1
    
    print("Collecting training pairs...")
    
    # Collect active consecutive pairs (X_n, X_{n+1})
    for t_idx in range(N_snap - 1):
        for particle_idx in range(Nsample):
            flag_n = exit_trajectory[particle_idx, t_idx]
            flag_n_plus_1 = exit_trajectory[particle_idx, t_idx + 1]
            
            # Both time steps must be active
            if flag_n == 1 and flag_n_plus_1 == 1:
                x1_n = sampled_trajectory_x1[particle_idx, t_idx]
                x2_n = sampled_trajectory_x2[particle_idx, t_idx]
                x1_n_plus_1 = sampled_trajectory_x1[particle_idx, t_idx + 1]
                x2_n_plus_1 = sampled_trajectory_x2[particle_idx, t_idx + 1]
                
                flow_x1.append(x1_n)
                flow_x2.append(x2_n)
                flow_y1.append(x1_n_plus_1)
                flow_y2.append(x2_n_plus_1)
    
    # Collect exit prediction pairs
    for t_idx in range(N_snap - 1):
        for particle_idx in range(Nsample):
            flag_n = exit_trajectory[particle_idx, t_idx]
            flag_n_plus_1 = exit_trajectory[particle_idx, t_idx + 1]
            
            if flag_n == 1:
                x1_n = sampled_trajectory_x1[particle_idx, t_idx]
                x2_n = sampled_trajectory_x2[particle_idx, t_idx]
                
                exit_x1.append(x1_n)
                exit_x2.append(x2_n)
                exit_delta.append(flag_n_plus_1)
    
    # Convert to numpy arrays
    flow_x = np.column_stack([np.array(flow_x1), np.array(flow_x2)])
    flow_y = np.column_stack([np.array(flow_y1), np.array(flow_y2)])
    exit_x = np.column_stack([np.array(exit_x1), np.array(exit_x2)])
    exit_delta = np.array(exit_delta)
    
    print(f"Collected {len(flow_x)} active consecutive pairs")
    print(f"Collected {len(exit_x)} exit prediction pairs")
    print(f"Exited particles: {np.sum(exit_delta==0)} out of {len(exit_x)} ({100*np.mean(exit_delta==0):.2f}%)")
    print(f"Remaining active: {np.sum(exit_delta==1)} out of {len(exit_x)} ({100*np.mean(exit_delta==1):.2f}%)")
    
    # Return trajectories for visualization
    trajectories = np.stack([sampled_trajectory_x1, sampled_trajectory_x2], axis=-1)
    
    return flow_x, flow_y, exit_x, exit_delta, trajectories


def visualize_data(flow_x, flow_y, exit_x, trajectories, figdir):
    """
    Create visualizations of the generated data
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Flow field visualization
    plt.subplot(2, 3, 1)
    plt.scatter(flow_x[:, 0], flow_x[:, 1], alpha=0.5, s=1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Flow Training Data Distribution')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Plot 2: Vector field at t=0
    plt.subplot(2, 3, 2)
    x1_grid = np.linspace(x_min, x_max, 20)
    x2_grid = np.linspace(y_min, y_max, 10)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    V1, V2 = velocity_field(X1, X2)
    plt.quiver(X1, X2, V1, V2, alpha=0.7)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Velocity Field at t=0')
    
    # Plot 3: Sample trajectories
    plt.subplot(2, 3, 3)
    n_traj_plot = min(100, trajectories.shape[0])
    for i in range(n_traj_plot):
        plt.plot(trajectories[i, :, 0], trajectories[i, :, 1], alpha=0.3, linewidth=0.5)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Sample Trajectories')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Plot 4: Flow displacement vectors
    plt.subplot(2, 3, 4)
    displacement = flow_y - flow_x
    plt.scatter(displacement[:, 0], displacement[:, 1], alpha=0.5, s=1)
    plt.xlabel('Δx1')
    plt.ylabel('Δx2')
    plt.title('Flow Displacements')
    
    # Plot 5: Exit prediction data
    plt.subplot(2, 3, 5)
    plt.scatter(exit_x[:, 0], exit_x[:, 1], alpha=0.5, s=1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Exit Prediction Data')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Plot 6: Trajectory time evolution
    plt.subplot(2, 3, 6)
    time_points = np.arange(trajectories.shape[1]) * sde_dt
    mean_x1 = np.mean(trajectories[:, :, 0], axis=0)
    mean_x2 = np.mean(trajectories[:, :, 1], axis=0)
    plt.plot(time_points, mean_x1, label='Mean x1')
    plt.plot(time_points, mean_x2, label='Mean x2')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Mean Trajectory Evolution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, 'data_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()


# Main execution
if __name__ == "__main__":
    
    Nsample = 200000  # Reduced for 2D due to computational cost

    print("=== 2D Advection-Diffusion SDE Data Generation ===")
    print(f"Domain: x1 ∈ [{x_min:.2f}, {x_max:.2f}], x2 ∈ [{y_min:.2f}, {y_max:.2f}]")
    print(f"Time: T = {sde_T}, dt = {sde_dt}")
    print(f"Parameters: Pe = {Pe}, ε = {epsilon}, n = {n}")
    print(f"Samples: {Nsample}")
    
    # Generate training data
    print("\nGenerating training data...")
    flow_x, flow_y, exit_x, exit_delta, trajectories = SDE_2d_advection_diffusion(
        sde_T, sde_dt, x_dim, Nsample, train=True
    )
    
    print(f"Generated {len(flow_x)} flow training pairs")
    print(f"Generated {len(exit_x)} exit prediction pairs")
    
    # Save training data
    print("\nSaving training data...")
    np.save(os.path.join(datadir, 'flow_x.npy'), flow_x)
    np.save(os.path.join(datadir, 'flow_y.npy'), flow_y)
    np.save(os.path.join(datadir, 'exit_x.npy'), exit_x)
    np.save(os.path.join(datadir, 'exit_delta.npy'), exit_delta)
    np.save(os.path.join(datadir, 'trajectories.npy'), trajectories)
    np.save(os.path.join(datadir, 'sde_dt.npy'), np.array([sde_dt]))
    
    # Save parameters
    params = {
        'Pe': Pe, 'epsilon': epsilon, 'n': n,
        'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max,
        'sde_T': sde_T, 'sde_dt': sde_dt, 'Nsample': Nsample
    }
    np.save(os.path.join(datadir, 'parameters.npy'), params)
    
    print("Training data saved successfully")
    
    # # Generate testing data
    # print("\nGenerating testing data...")
    # _, _, _, _, test_trajectories = SDE_2d_advection_diffusion(
    #     sde_T, sde_dt, x_dim, 1000, train=False  # Smaller test set
    # )
    
    # np.save(os.path.join(datadir, 'test_trajectories.npy'), test_trajectories)
    # print("Testing data saved successfully")
    
    # # Create visualizations
    # print("\nGenerating visualizations...")
    # visualize_data(flow_x, flow_y, exit_x, trajectories, figdir)
    # print("Visualizations saved successfully")
    
    # print("\n=== Data Generation Complete ===")
    # print(f"Data saved in: {datadir}")
    # print(f"Figures saved in: {figdir}")