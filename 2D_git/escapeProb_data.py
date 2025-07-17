import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

datadir = os.path.join(f'data_2d')
figdir = os.path.join(f'fig_2d')

# 2D domain parameters
x_min, x_max = -np.pi, np.pi  # x1 domain
y_min, y_max = 0.0, 2.0       # x2 domain

# Physical parameters
Pe = 5.
epsilon = 0.0

n = 2

def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def velocity_field(x1, x2):
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
    Check if particles have exited the domain by reaching x2=0 or x2=2 boundaries
    Returns flag array: 1 for still inside domain, 0 for exited
    """
    flag = np.ones(len(x1))
    
    # Exit condition: particle reaches top (x2 >= y_max) or bottom (x2 <= y_min) boundary
    exited = (x2 >= y_max) | (x2 <= y_min)
    flag[exited] = 0
    
    return flag

def SDE_2d_advection_diffusion_batch(x1_ini, x2_ini, T, mc_dt):
    """
    2D Advection-Diffusion SDE solver for a batch of initial conditions
    Returns final flags indicating which particles survived
    """
    Nsample = len(x1_ini)
    Nt = int(np.floor(T / mc_dt) + 1)
    
    # Current positions
    x1_end = np.copy(x1_ini)
    x2_end = np.copy(x2_ini)
    flag = np.ones(Nsample)
    n_flag = Nsample
    
    for ii in range(Nt):
        if n_flag <= 0:
            break
        
        # Get active particles
        idx_flag = np.where(flag == 1)[0]
        n_active = len(idx_flag)
        
        if n_active == 0:
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
    
    return flag

def RE_2d_escape_single_time(target_time, dim, num_points=101, num_realizations=1000, batch_size=10000):
    """
    Generate 2D Monte Carlo escape data for a single target time point.
    
    Parameters:
    target_time: float, the time at which to sample (e.g., 1.0)
    dim: int, dimension of the system (2)
    num_points: int, number of grid points per axis (101)
    num_realizations: int, number of MC realizations per grid point (1000)
    batch_size: int, batch size for processing
    """
    
    # Ensure data directory exists
    make_folder(datadir)
    
    # Define parameter ranges for 2D grid
    x1_vals = np.linspace(x_min, x_max, num_points)
    x2_vals = np.linspace(y_min, y_max, num_points)
    
    # Create meshgrid
    X1, X2 = np.meshgrid(x1_vals, x2_vals, indexing='ij')
    
    # Flatten and stack into a 2D array - these are the unique grid points
    grid_points = np.stack([X1.ravel(), X2.ravel()], axis=1)
    num_grid_points = grid_points.shape[0]
    
    # Create repeated initial conditions: each grid point repeated num_realizations times
    AllY0_data = np.repeat(grid_points, num_realizations, axis=0)
    
    total_samples = AllY0_data.shape[0]
    num_batches = int(np.ceil(total_samples / batch_size))
    
    print(f"Generating 2D MC escape data for time = {target_time}")
    print(f"Grid resolution: {num_points}² = {num_grid_points} unique points")
    print(f"MC realizations per point: {num_realizations}")
    print(f"Total samples: {total_samples} ({num_grid_points} × {num_realizations})")
    print(f"Processing in {num_batches} batches of size {batch_size}")
    print(f"Domain: x1 ∈ [{x_min:.2f}, {x_max:.2f}], x2 ∈ [{y_min:.2f}, {y_max:.2f}]")
    
    # Monte Carlo time step
    mc_dt = 0.0005  # Fine time step for numerical integration
    nt = int(target_time / mc_dt)
    print(f"Running {nt} Monte Carlo steps (dt = {mc_dt})")
    
    all_data_batches = []
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(total_samples, (batch_idx + 1) * batch_size)
        
        # Initial conditions for this batch
        x_ini = AllY0_data[start_idx:end_idx].copy()
        x1_ini = x_ini[:, 0]
        x2_ini = x_ini[:, 1]
        
        current_batch_size = x_ini.shape[0]
        
        # Run Monte Carlo simulation for this batch
        final_flags = SDE_2d_advection_diffusion_batch(x1_ini, x2_ini, target_time, mc_dt)
        
        # Store results for this batch
        # Create a 2D array with shape (1, current_batch_size) to match expected format
        flags_2d = final_flags.reshape(1, -1)  # Shape: (1, batch_size)
        
        all_data_batches.append({
            'flags': flags_2d,
            'initial': x_ini
        })
        
        print(f"Batch {batch_idx+1}: {np.sum(final_flags)} particles remaining out of {current_batch_size}")
    
    # Create the final data structure matching the expected format
    times_array = np.array([target_time])  # Single time point
    
    combined_result = {
        'sampled_data': all_data_batches,
        'times': times_array,
        'grid_info': {
            'num_points': num_points,
            'num_realizations': num_realizations,
            'num_grid_points': num_grid_points
        }
    }
    
    # Save the results
    filename = f'escape_2d_true_datanx{num_points}_nr{num_realizations}.npy'
    filepath = os.path.join(datadir, filename)
    np.save(filepath, combined_result)
    
    print(f"Data saved to: {filepath}")
    
    # Calculate and print statistics
    total_flags = np.concatenate([batch['flags'] for batch in all_data_batches], axis=1)
    
    # Reshape to (num_grid_points, num_realizations) for analysis
    flags_reshaped = total_flags.reshape(num_grid_points, num_realizations)
    
    # Calculate escape probability for each grid point (average over realizations)
    escape_probabilities = 1 - np.mean(flags_reshaped, axis=1)  # Average survival rate per grid point
    overall_survival_rate = np.mean(total_flags)
    overall_escape_rate = 1 - overall_survival_rate
    
    print(f"Final statistics:")
    print(f"  Overall survival rate: {overall_survival_rate:.4f}")
    print(f"  Overall escape rate: {overall_escape_rate:.4f}")
    print(f"  Total samples: {total_flags.shape[1]}")
    print(f"  Grid points: {num_grid_points}")
    print(f"  Realizations per point: {num_realizations}")
    print(f"  Min escape probability: {escape_probabilities.min():.4f}")
    print(f"  Max escape probability: {escape_probabilities.max():.4f}")
    print(f"  Mean escape probability: {escape_probabilities.mean():.4f}")
    
    return combined_result

def generate_2d_escape_probabilities(target_time=1.0, dim=2):
    """
    Generate 2D Monte Carlo escape data for multiple grid resolutions.
    """
    print("=" * 60)
    print("2D MONTE CARLO ESCAPE DATA GENERATION")
    print("=" * 60)
    print(f"Physical parameters:")
    print(f"  Pe = {Pe}")
    print(f"  epsilon = {epsilon}")
    print(f"  n = {n}")
    print(f"  Domain: x1 ∈ [{x_min:.2f}, {x_max:.2f}], x2 ∈ [{y_min:.2f}, {y_max:.2f}]")
    
    # Generate data for nx=101 (high resolution)
    print("\n" + "="*40)
    print("GENERATING HIGH RESOLUTION DATA (nx=101)")
    print("="*40)
    result_101 = RE_2d_escape_single_time(target_time, dim, num_points=101,num_realizations=2000, batch_size=50000)
    
    # Generate data for nx=31 (medium resolution)
    print("\n" + "="*40)
    print("GENERATING MEDIUM RESOLUTION DATA (nx=31)")
    print("="*40)
    result_31 = RE_2d_escape_single_time(target_time, dim, num_points=31, num_realizations=1000, batch_size=50000)
    
    print("\n" + "="*40)
    print("2D DATA GENERATION COMPLETED")
    print("="*40)
    print(f"Files created:")
    print(f"  - {datadir}/escape_2d_true_datanx101.npy")
    print(f"  - {datadir}/escape_2d_true_datanx31.npy")
    print(f"Target time: {target_time}")
    
    return result_101, result_31

def load_and_analyze_2d_data(filename):
    """
    Load and analyze the generated 2D escape data
    """
    filepath = os.path.join(datadir, filename)
    try:
        data = np.load(filepath, allow_pickle=True).item()
        
        # Extract information
        sampled_batches = data['sampled_data']
        time_points = data['times']
        grid_info = data.get('grid_info', {})
        
        # Concatenate all flags
        all_flags = np.concatenate([b['flags'] for b in sampled_batches], axis=1)
        
        num_points = grid_info.get('num_points', 'unknown')
        num_realizations = grid_info.get('num_realizations', 'unknown')
        num_grid_points = grid_info.get('num_grid_points', 'unknown')
        
        print(f"Loaded data from: {filename}")
        print(f"Time points: {time_points}")
        print(f"Grid points: {num_grid_points} ({num_points}×{num_points})")
        print(f"Realizations per point: {num_realizations}")
        print(f"Total samples: {all_flags.shape[1]}")
        print(f"Overall survival rate: {np.mean(all_flags):.4f}")
        print(f"Overall escape rate: {1 - np.mean(all_flags):.4f}")
        
        return data
        
    except FileNotFoundError:
        print(f"File {filepath} not found. Please generate the data first.")
        return None

# Main execution
if __name__ == "__main__":
    # Set parameters
    dim = 2
    target_time = 0.05  # Target time for escape probability calculation
    
    print("Starting 2D Monte Carlo escape probability generation...")
    print(f"Target time: {target_time}")
    print(f"MC time step: 0.0005")
    
    # Generate data for both resolutions
    result_101, result_31 = generate_2d_escape_probabilities(target_time, dim)
    