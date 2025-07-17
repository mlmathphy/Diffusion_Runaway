import torch
import numpy as np
import os
from scipy.stats import norm
from tqdm import tqdm
from scipy.io import loadmat

datadir = os.path.join('data')

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

def RE_3dfun_single_time(target_time, dim, num_points=101, batch_size=100000):
    """
    Generate Monte Carlo data for a single target time point.
    
    Parameters:
    target_time: float, the time at which to sample (e.g., 0.2)
    dim: int, dimension of the system (3)
    num_points: int, number of grid points per axis (101)
    batch_size: int, batch size for processing
    """
    
    # Ensure data directory exists
    make_folder(datadir)
    
    # Define parameter ranges
    r_max, p_min, p_max, xi_min, xi_max = 1, 0.5, 5, -1, 1
    mc_dt = 0.0005  # Monte Carlo time step
    
    # Generate the mesh grid for initial conditions
    p1 = np.linspace(p_min, p_max, num_points)
    xi1 = np.linspace(xi_min, xi_max, num_points)
    r1 = np.linspace(0.8, r_max, num_points)
    
    # Create meshgrid in 'ij' indexing to preserve the input order
    P, XI, R = np.meshgrid(p1, xi1, r1, indexing='ij')
    
    # Flatten and stack into a 2D array
    AllY0_data = np.stack([P.ravel(), XI.ravel(), R.ravel()], axis=1)
    
    total_samples = AllY0_data.shape[0]
    num_batches = int(np.ceil(total_samples / batch_size))
    
    print(f"Generating MC data for time = {target_time}")
    print(f"Processing {total_samples} samples in {num_batches} batches")
    print(f"Grid resolution: {num_points}³ = {num_points**3} points")
    
    # Calculate number of MC steps needed
    nt = int(target_time / mc_dt)
    print(f"Running {nt} Monte Carlo steps (dt = {mc_dt})")
    
    all_data_batches = []
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(total_samples, (batch_idx + 1) * batch_size)
        
        # Initial conditions for this batch
        x_ini = AllY0_data[start_idx:end_idx].copy()
        x_end = x_ini.copy()
        
        current_batch_size = x_ini.shape[0]
        
        # Initialize flags (1 = particle still in domain, 0 = escaped)
        flag = np.ones(current_batch_size, dtype=int)
        n_flag = current_batch_size
        
        # Run Monte Carlo simulation
        for i in tqdm(range(nt), desc=f"MC steps for batch {batch_idx+1}", leave=False):
            if n_flag <= 0:
                break
                
            # Find particles still in domain
            idx_flag = np.where(flag == 1)[0]
            if len(idx_flag) == 0:
                break
                
            x_temp = x_end[idx_flag]
            
            # Calculate drift and diffusion
            drift_dt = drift_3d(x_temp)
            diff_dt = diffusion_3d(x_temp)
            
            # Generate random increments
            Wt = np.random.normal(0, np.sqrt(mc_dt), (len(idx_flag), dim))
            
            # Update positions
            x_end[idx_flag] += drift_dt * mc_dt + diff_dt * Wt
            
            # Apply boundary conditions
            # Reflect r at r=0
            x_end[idx_flag, 2] = np.abs(x_end[idx_flag, 2])
            
            # Reflect p at p_min
            x_end[x_end[:, 0] < p_min, 0] = p_min
            
            # Reflect xi at boundaries
            mask_xi_high = x_end[:, 1] >= 1
            x_end[mask_xi_high, 1] = 2 - x_end[mask_xi_high, 1]
            
            mask_xi_low = x_end[:, 1] <= -1
            x_end[mask_xi_low, 1] = -2 - x_end[mask_xi_low, 1]
            
            # Check for escape (r >= r_max)
            escaped_particles = np.where(x_end[:, 2] >= r_max)[0]
            flag[escaped_particles] = 0
            n_flag = np.sum(flag)
        
        # Store results for this batch
        # Create a 2D array with shape (1, current_batch_size) to match expected format
        flags_2d = flag.reshape(1, -1)  # Shape: (1, batch_size)
        
        all_data_batches.append({
            'flags': flags_2d,
            'initial': x_ini
        })
        
        print(f"Batch {batch_idx+1}: {np.sum(flag)} particles remaining out of {current_batch_size}")
    
    # Create the final data structure matching the expected format
    times_array = np.array([target_time])  # Single time point
    
    combined_result = {
        'sampled_data': all_data_batches,
        'times': times_array
    }
    
    # Save the results
    filename = f'escape_true_datanx{num_points}.npy'
    filepath = os.path.join(datadir, filename)
    np.save(filepath, combined_result)
    
    print(f"Data saved to: {filepath}")
    
    # Print statistics
    total_flags = np.concatenate([batch['flags'] for batch in all_data_batches], axis=1)
    survival_rate = np.mean(total_flags)
    escape_rate = 1 - survival_rate
    
    print(f"Final statistics:")
    print(f"  Survival rate: {survival_rate:.4f}")
    print(f"  Escape rate: {escape_rate:.4f}")
    print(f"  Total particles: {total_flags.shape[1]}")
    print(f"  Survived: {np.sum(total_flags)}")
    print(f"  Escaped: {np.sum(1 - total_flags)}")
    
    return combined_result

def generate_multiple_resolutions(target_time=0.2, dim=3):
    """
    Generate Monte Carlo data for multiple grid resolutions.
    """
    print("=" * 60)
    print("MONTE CARLO DATA GENERATION")
    print("=" * 60)
    
    # Generate data for nx=101 (high resolution)
    print("\n" + "="*40)
    print("GENERATING HIGH RESOLUTION DATA (nx=101)")
    print("="*40)
    result_101 = RE_3dfun_single_time(target_time, dim, num_points=101, batch_size=50000)
    
    # Generate data for nx=31 (medium resolution)
    print("\n" + "="*40)
    print("GENERATING MEDIUM RESOLUTION DATA (nx=31)")
    print("="*40)
    result_31 = RE_3dfun_single_time(target_time, dim, num_points=31, batch_size=100000)
    
    print("\n" + "="*40)
    print("DATA GENERATION COMPLETED")
    print("="*40)
    print(f"Files created:")
    print(f"  - data/escape_true_datanx101.npy")
    print(f"  - data/escape_true_datanx31.npy")
    print(f"Target time: {target_time}")
    
    return result_101, result_31

# Main execution
if __name__ == "__main__":
    # Set parameters
    dim = 3
    target_time = 0.2  # Delta t = 0.2 as requested
    
    # Generate data for both resolutions
    generate_multiple_resolutions(target_time, dim)