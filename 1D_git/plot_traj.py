import numpy as np
import matplotlib.pyplot as plt

def simulate_brownian_motion(T=3.0, dt=0.01, DT=0.1, n_trajectories=1000, x_range=[3, 3], x_min=0, x_max=5):
    """
    Simulate Brownian motion trajectories X_t = W_t with absorbing boundaries
    
    Parameters:
    - T: Total simulation time
    - dt: Small time step for simulation accuracy
    - DT: Large time step for data recording/plotting
    - n_trajectories: Number of trajectories to simulate
    - x_range: Range for initial positions [min, max] (set to [3,3] for all starting at x=3)
    - x_min: Lower boundary (absorbing)
    - x_max: Upper boundary (absorbing)
    
    Returns:
    - t_record: Time array for recorded data points
    - trajectories: Array of recorded trajectories (n_trajectories x n_record_steps)
    - exit_flags: Array of exit flags for each recorded point (1=active, 0=exited)
    - flow_x, flow_y: Active consecutive pairs for flow analysis
    - exit_x, exit_delta: Exit prediction pairs (position, next_flag)
    """
    
    # Time parameters for simulation
    n_sim_steps = int(T / dt) + 1
    t_sim = np.linspace(0, T, n_sim_steps)
    
    # Time parameters for recording
    n_record_steps = int(T / DT) + 1
    t_record = np.linspace(0, T, n_record_steps)
    record_interval = int(DT / dt)  # How many simulation steps between recordings
    
    print(f"Simulation steps: {n_sim_steps}, Recording steps: {n_record_steps}")
    print(f"Recording every {record_interval} simulation steps")
    
    # Initialize trajectories and tracking arrays
    trajectories = np.zeros((n_trajectories, n_record_steps))
    exit_flags = np.zeros((n_trajectories, n_record_steps))  # 1=active, 0=exited
    
    # All trajectories start at x=3
    x_initial = np.full(n_trajectories, x_range[0], dtype=float)  # All start at x=3
    trajectories[:, 0] = x_initial  # Record initial positions
    exit_flags[:, 0] = 1  # All start active
    
    # Track which trajectories are still active
    flag = np.ones(n_trajectories, dtype=bool)  # True = active, False = exited
    exit_times = np.full(n_trajectories, np.nan)  # Time when trajectory exits
    exit_positions = np.full(n_trajectories, np.nan)  # Position where trajectory exits
    
    # Current positions
    x_current = np.copy(x_initial)
    
    # Index for recording
    record_idx = 1
    
    # Generate Brownian motion increments
    for i in range(1, n_sim_steps):
        # Only update active trajectories
        active_idx = np.where(flag)[0]
        n_active = len(active_idx)
        
        if n_active == 0:
            print(f"All trajectories exited by simulation step {i}")
            break
            
        # Brownian motion increments for active trajectories
        dW = np.random.normal(0, np.sqrt(dt), n_active)
        
        # Update positions for active trajectories
        x_current[active_idx] += dW
        
        # Check boundary conditions
        row_out = np.where((x_current >= x_max) | (x_current <= x_min))[0]
        
        if len(row_out) > 0:
            # Mark trajectories that exceed domain as inactive
            flag[row_out] = False
            # Record exit times and positions
            exit_times[row_out] = t_sim[i]
            exit_positions[row_out] = x_current[row_out]
            if i % 1000 == 0:  # Print less frequently for 1000 trajectories
                print(f"Sim step {i}: {len(row_out)} trajectories exited. Remaining active: {np.sum(flag)}")
        
        # Record data at specified intervals
        if i % record_interval == 0 and record_idx < n_record_steps:
            # Store current positions and flags for all trajectories
            for j in range(n_trajectories):
                if flag[j]:  # Still active
                    trajectories[j, record_idx] = x_current[j]
                    exit_flags[j, record_idx] = 1
                else:  # Exited - keep at exit position
                    trajectories[j, record_idx] = exit_positions[j]
                    exit_flags[j, record_idx] = 0
            record_idx += 1
    
    # Fill remaining time points for exited trajectories
    for record_idx in range(record_idx, n_record_steps):
        for j in range(n_trajectories):
            if not flag[j]:  # Exited trajectory
                trajectories[j, record_idx] = exit_positions[j]
                exit_flags[j, record_idx] = 0
            else:  # Still active
                trajectories[j, record_idx] = x_current[j]
                exit_flags[j, record_idx] = 1
    
    # Collect active consecutive pairs (X_n, X_{n+1}) - both points active
    flow_x = []
    flow_y = []
    
    # Collect exit prediction pairs (X_n, flag_{n+1}) - first point active
    exit_x = []
    exit_delta = []
    
    for t_idx in range(n_record_steps - 1):
        for particle_idx in range(n_trajectories):
            flag_n = exit_flags[particle_idx, t_idx]        # flag at time n
            flag_n_plus_1 = exit_flags[particle_idx, t_idx + 1]  # flag at time n+1
            
            # Active consecutive pairs: both active
            if flag_n == 1 and flag_n_plus_1 == 1:
                x_n = trajectories[particle_idx, t_idx]
                x_n_plus_1 = trajectories[particle_idx, t_idx + 1]
                flow_x.append(x_n)
                flow_y.append(x_n_plus_1)
            
            # Exit prediction pairs: first point active
            if flag_n == 1:
                x_n = trajectories[particle_idx, t_idx]
                exit_x.append(x_n)
                exit_delta.append(flag_n_plus_1)
    
    return t_record, trajectories, exit_flags, flow_x, flow_y, exit_x, exit_delta

def analyze_exits_at_times(t, exit_flags, target_times=[1.0, 2.0, 3.0]):
    """
    Analyze how many particles have exited at specific time points
    
    Parameters:
    - t: Time array
    - exit_flags: Exit flags array (n_trajectories x n_time_steps)
    - target_times: List of times to analyze
    
    Returns:
    - exit_counts: Dictionary with time as key and exit count as value
    - survival_counts: Dictionary with time as key and survival count as value
    """
    exit_counts = {}
    survival_counts = {}
    
    for target_time in target_times:
        # Find the closest time index
        time_idx = np.argmin(np.abs(t - target_time))
        actual_time = t[time_idx]
        
        # Count exited particles (flag = 0) at this time
        n_exited = np.sum(exit_flags[:, time_idx] == 0)
        n_survived = np.sum(exit_flags[:, time_idx] == 1)
        
        exit_counts[actual_time] = n_exited
        survival_counts[actual_time] = n_survived
        
        print(f"At t = {actual_time:.1f}: {n_exited} exited, {n_survived} survived")
    
    return exit_counts, survival_counts

def plot_trajectories_with_exit_analysis(t, trajectories, exit_flags, x_min=0, x_max=5):
    """
    Plot subset of trajectories and show exit analysis
    """
    from scipy.stats import gaussian_kde
    
    # Create three subplots in one row with different widths
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), 
                                        gridspec_kw={'width_ratios': [3, 1, 1.5]})
    
    # Plot 1: Sample of trajectories (plot only first 20 for visibility)
    n_plot = min(20, trajectories.shape[0])
    
    for i in range(n_plot):
        # Find where trajectory becomes inactive (exits)
        active_points = np.where(exit_flags[i] == 1)[0]
        inactive_points = np.where(exit_flags[i] == 0)[0]
        
        if len(active_points) > 0:
            if len(inactive_points) > 0:
                # Trajectory exits - plot until first exit point (inclusive)
                first_exit_idx = inactive_points[0]
                # Plot from start to first exit point
                t_plot = t[:first_exit_idx + 1]
                x_plot = trajectories[i, :first_exit_idx + 1]
                ax1.plot(t_plot, x_plot, color='r', alpha=0.5, linewidth=2)
            else:
                # Trajectory never exits - plot all active points
                t_active = t[active_points]
                x_active = trajectories[i, active_points]
                ax1.plot(t_active, x_active, color='b', alpha=0.5, linewidth=2)
    
    # Add boundary lines
    ax1.axhline(y=x_min, color='k', linestyle='--', alpha=0.7, linewidth=2, label='Boundaries')
    ax1.axhline(y=x_max, color='k', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add vertical lines at analysis times
    for target_time in [1.0, 2.0, 3.0]:
        ax1.axvline(x=target_time, color='k', linestyle=':', alpha=0.8, linewidth=2)
        ax1.text(target_time, x_max + 0.2, f't={target_time}', 
                ha='center', fontsize=12, color='k')
    
    ax1.set_xlabel('Time (t)', fontsize=16)
    ax1.set_ylabel('Position X(t)', fontsize=16)
    ax1.set_title(f'Sample of {n_plot} Trajectories', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(x_min - 0.5, x_max + 0.5)
    ax1.legend()

       # Plot 2: Histogram of non-exited particles at t=3
    # Find time index closest to t=3
    t3_idx = np.argmin(np.abs(t - 3.0))
    actual_t3 = t[t3_idx]
    
    # Get positions of non-exited particles at t=3
    non_exited_mask = exit_flags[:, t3_idx] == 1
    non_exited_positions = trajectories[non_exited_mask, t3_idx]
    
    if len(non_exited_positions) > 1:
        # Plot histogram horizontally (counts on x-axis, position on y-axis)
        counts, bins, patches = ax2.hist(non_exited_positions, bins=20, orientation='horizontal', 
                                        alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    else:
        ax2.text(0.5, 2.5, 'No survivors\nat t=3', ha='center', va='center', 
                fontsize=12, transform=ax2.transData)
    
    ax2.set_xlabel('Count', fontsize=16)
    ax2.set_ylabel('Position X', fontsize=16)
    ax2.set_title(f'Distribution\n(Non-exited at t=3)', fontsize=16)
    ax2.set_ylim(x_min - 0.5, x_max + 0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add boundary lines to histogram plot
    ax2.axhline(y=x_min, color='k', linestyle='--', alpha=0.5)
    ax2.axhline(y=x_max, color='k', linestyle='--', alpha=0.5)
    
    
    # Plot 3: Exit analysis
    exit_counts, survival_counts = analyze_exits_at_times(t, exit_flags)
    
    times = list(exit_counts.keys())
    exits = list(exit_counts.values())
    survivals = list(survival_counts.values())
    
    # Bar plot showing exits vs survivals
    width = 0.35
    x_pos = np.arange(len(times))
    
    bars1 = ax3.bar(x_pos - width/2, exits, width, label='Exited', color='red', alpha=0.7)
    bars2 = ax3.bar(x_pos + width/2, survivals, width, label='Non-exited', color='green', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax3.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 10,
                f'{exits[i]}', ha='center', va='bottom', fontsize=10)
        ax3.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 10,
                f'{survivals[i]}', ha='center', va='bottom', fontsize=10)
    
    ax3.set_xlabel('Time', fontsize=16)
    ax3.set_ylabel('Number of Particles', fontsize=16)
    ax3.set_title('Exited vs Non-exited', fontsize=16)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f't={t:.1f}' for t in times])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # # Add total count annotation
    # total_particles = trajectories.shape[0]
    # ax3.text(0.02, 0.98, f'Total: {total_particles}', 
    #         transform=ax3.transAxes, fontsize=10, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('exit_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print KDE statistics
    if len(non_exited_positions) > 1:
        print(f"\nKDE Analysis at t=3:")
        print(f"Non-exited particles: {len(non_exited_positions)}")
        print(f"Mean position: {np.mean(non_exited_positions):.3f}")
        print(f"Std deviation: {np.std(non_exited_positions):.3f}")
        print(f"Min position: {np.min(non_exited_positions):.3f}")
        print(f"Max position: {np.max(non_exited_positions):.3f}")
    
    return exit_counts, survival_counts

def main():
    """
    Main function to run the simulation and create exit analysis
    """
    # Set random seed for reproducibility (optional)
    # np.random.seed(42)
    
    # Simulation parameters
    T = 3.0          # Total time
    dt = 0.001        # Small time step for simulation accuracy
    DT = 0.1         # Large time step for data recording/plotting
    n_traj = 100000    # Number of trajectories
    x_range = [1,1] # All start at x=3
    x_min = 0        # Lower boundary
    x_max = 6        # Upper boundary
    
    print(f"Simulating {n_traj} Brownian motion trajectories with absorbing boundaries...")
    print(f"All trajectories start at x = {x_range[0]}")
    print(f"Time span: 0 to {T}")
    print(f"Simulation time step (dt): {dt}")
    print(f"Recording time step (DT): {DT}")
    print(f"Boundaries: [{x_min}, {x_max}]")
    
    # Run simulation
    time, trajectories, exit_flags, flow_x, flow_y, exit_x, exit_delta = simulate_brownian_motion(
        T, dt, DT, n_traj, x_range, x_min, x_max)
    
    # Display final statistics
    print(f"\nSimulation completed!")
    
    # Count trajectories that exited vs Non-exited at final time
    final_flags = exit_flags[:, -1]
    n_exited_final = np.sum(final_flags == 0)
    n_survived_final = np.sum(final_flags == 1)
    
    print(f"\nFinal Results (at t={T}):")
    print(f"Trajectories that hit boundaries: {n_exited_final}")
    print(f"Trajectories that survived: {n_survived_final}")
    print(f"Exit percentage: {100 * n_exited_final / n_traj:.1f}%")
    
    # Create plot with exit analysis
    print("\nAnalyzing exits at t=1, 2, 3...")
    exit_counts, survival_counts = plot_trajectories_with_exit_analysis(
        time, trajectories, exit_flags, x_min, x_max)
    
    return time, trajectories, exit_flags, exit_counts, survival_counts

# Run the simulation
if __name__ == "__main__":
    t, traj, flags, exit_counts, survival_counts = main()