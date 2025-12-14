import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION & PHYSICS CONSTANTS ---
MASS = 100.0          # kg (Spacecraft Mass)
DT = 1.0              # s (Time step)
SIM_DURATION = 2000   # s (Total simulation time)
STEPS = int(SIM_DURATION / DT)

# DISTURBANCE (Solar Wind / Drag)
DISTURBANCE_MEAN = 0.0001 # Constant drift force (N)
DISTURBANCE_STD = 0.0005  # Stochastic noise (N)

# AGENT 1: CLASSIC (Chemical Deadband)
DEADBAND_LIMIT = 2.0  # meters (Allowed drift before firing)
THRUST_CHEM = 1.0     # N (Chemical Thruster force - Impulsive)

# AGENT 2: IEN (SRP / Continuous)
# NOTE: Gains represent the minimization of Information Divergence
KP_IEN = 0.05         # Proportional Gain (Position Error)
KV_IEN = 1.0          # Derivative Gain (Velocity Error)
MAX_SRP_FORCE = 0.005 # N (Max force available from Solar Radiation Pressure)

def simulate():
    # Time array
    t = np.arange(0, SIM_DURATION, DT)
    
    # --- INITIALIZATION ---
    # State vectors: [Position, Velocity]
    x_classic = np.zeros(STEPS)
    v_classic = np.zeros(STEPS)
    dv_classic = np.zeros(STEPS) # Cumulative Delta-V
    
    x_ien = np.zeros(STEPS)
    v_ien = np.zeros(STEPS)
    dv_ien = np.zeros(STEPS)     # Cumulative Delta-V (Chemical)

    # Initial Conditions
    x_classic[0] = 0.5
    x_ien[0] = 0.5

    np.random.seed(42) # For reproducible noise
    noise = np.random.normal(DISTURBANCE_MEAN, DISTURBANCE_STD, STEPS)

    # --- SIMULATION LOOP ---
    for i in range(STEPS - 1):
        
        # 1. PHYSICS ENGINE (Environment)
        # Apply external disturbance force (F = ma -> a = F/m)
        a_env = noise[i] / MASS
        
        # --- AGENT 1: CLASSIC DEADBAND ---
        u_classic = 0.0
        # Logic: If error exceeds deadband, fire thruster to oppose it
        if abs(x_classic[i]) > DEADBAND_LIMIT:
            direction = -np.sign(x_classic[i])
            u_classic = THRUST_CHEM * direction
            # Accumulate Propellant Usage (Delta-V)
            # dv = F/m * dt
            dv_classic[i+1] = dv_classic[i] + (abs(u_classic) / MASS * DT)
        else:
            dv_classic[i+1] = dv_classic[i]

        # Integrate Classic State (Euler Integration)
        a_total_classic = a_env + (u_classic / MASS)
        v_classic[i+1] = v_classic[i] + a_total_classic * DT
        x_classic[i+1] = x_classic[i] + v_classic[i] * DT


        # --- AGENT 2: IEN (Information-Entropic / SRP) ---
        # Logic: Continuous correction minimizing divergence (Proportional-Derivative)
        # u_req is the "Required Force" to minimize entropy rate
        u_req = -KP_IEN * x_ien[i] - KV_IEN * v_ien[i]
        
        # Constraint: Can only use available SRP (Solar Radiation Pressure)
        # We clip the force to the physical limits of the solar sail/vanes
        u_ien = np.clip(u_req, -MAX_SRP_FORCE, MAX_SRP_FORCE)
        
        # Propellant Usage: IEN uses SRP (Environment), so Chemical Delta-V is ZERO.
        dv_ien[i+1] = dv_ien[i] # No change

        # Integrate IEN State
        a_total_ien = a_env + (u_ien / MASS)
        v_ien[i+1] = v_ien[i] + a_total_ien * DT
        x_ien[i+1] = x_ien[i] + v_ien[i] * DT

    return t, x_classic, dv_classic, x_ien, dv_ien

def plot_results(t, x_c, dv_c, x_i, dv_i):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: Position Holding
    ax1.set_title("Station Keeping Performance: Reactive vs. Entropic", fontsize=14)
    ax1.plot(t, x_c, 'r-', alpha=0.6, label='Classic Deadband (Chemical)')
    ax1.plot(t, x_i, 'b-', linewidth=2, label='IEN Agent (SRP / Active Inference)')
    ax1.axhline(y=2.0, color='k', linestyle='--', alpha=0.3, label='Deadband Limit')
    ax1.axhline(y=-2.0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel("Position Error (m)", fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative Delta-V
    ax2.plot(t, dv_c, 'r-', alpha=0.6, label='Fuel Consumed (Classic)')
    ax2.plot(t, dv_i, 'b-', linewidth=2, label='Fuel Consumed (IEN)')
    ax2.set_ylabel("Cumulative Delta-V (m/s)", fontsize=12)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Annotations
    final_dv = dv_c[-1]
    ax2.text(t[-1], final_dv, f" Total Δv: {final_dv:.2f} m/s", color='red', verticalalignment='bottom')
    ax2.text(t[-1], 0, f" Total Δv: 0.00 m/s", color='blue', verticalalignment='bottom')

    plt.tight_layout()
    
    # SAVE THE KILL SHOT
    filename = "IEN_vs_Classic_Comparison.png"
    plt.savefig(filename, dpi=300)
    print(f"✅ Simulation Complete. Graph saved as: {filename}")
    plt.show()

if __name__ == "__main__":
    print("🚀 Initializing IEN vs Classic Simulation...")
    t, x_c, dv_c, x_i, dv_i = simulate()
    plot_results(t, x_c, dv_c, x_i, dv_i)