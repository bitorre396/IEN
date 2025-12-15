import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
IEN PROJECT - PHASE 2: CR3BP DYNAMICS & ENTROPIC CONTROL
--------------------------------------------------------
Branch: dev-cr3bp
Context: Circular Restricted 3-Body Problem (Earth-Moon)
Objective: Stabilization at Unstable Lagrange Point (L1) via Active Inference.

NOTE ON PHYSICS:
This simulation performs a 'Saddle Point Stabilization' test.
The agent is tasked with 'hovering' at the exact L1 point, which is dynamically 
unstable. This is a stress-test for control authority, distinct from following 
a natural periodic Halo orbit (which would require less energy).
"""

# --- CONSTANTS (Earth-Moon System) ---
MU = 0.0121505856  # Mass parameter (Moon mass / Total mass)
L1_POS = 0.836915  # Approx L1 location (dimensionless)

# --- CR3BP PHYSICS ENGINE ---
def equations_of_motion(t, state, control_u):
    """
    Circular Restricted 3-Body Problem Equations
    State: [x, y, z, vx, vy, vz]
    Reference: Szebehely, V. (1967). Theory of orbits.
    """
    x, y, z, vx, vy, vz = state
    
    # Distances to Primaries
    r1 = np.sqrt((x + MU)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + MU)**2 + y**2 + z**2)
    
    # Gravitational Potential Gradients (Omega)
    Ox = x - (1-MU)*(x+MU)/r1**3 - MU*(x-1+MU)/r2**3
    Oy = y - (1-MU)*y/r1**3 - MU*y/r2**3
    Oz =   - (1-MU)*z/r1**3 - MU*z/r2**3
    
    # Accelerations (inc. Coriolis 2*vy, -2*vx)
    ax = Ox + 2*vy + control_u[0]
    ay = Oy - 2*vx + control_u[1]
    az = Oz        + control_u[2]
    
    return [vx, vy, vz, ax, ay, az]

# --- IEN AGENT (Active Inference Controller) ---
class IENAgent:
    def __init__(self, target_state):
        self.target = np.array(target_state)
        # Precision Matrices (Inverse Covariance)
        # These represent the agent's 'confidence' in its sensors vs model.
        # High precision = High control gain (Thermodynamic equivalent).
        self.Pi_x = np.diag([5000, 5000, 5000]) 
        self.Pi_v = np.diag([10000, 10000, 10000])    
    
    def get_control(self, current_state):
        """
        Thermodynamic Unification:
        Control u = - Gradient(Variational Free Energy) 
        
        In this linear-gaussian approximation, the control law emerges as:
        u = - (Precision_Position * Error_Position + Precision_Velocity * Error_Velocity)
        """
        curr = np.array(current_state)
        
        # Error calculation (Divergence from Setpoint)
        e_pos = curr[:3] - self.target[:3]
        e_vel = curr[3:] - self.target[3:]
        
        # Entropic Control Force
        u_x = -self.Pi_x @ e_pos
        u_v = -self.Pi_v @ e_vel
        
        u_total = u_x + u_v
        
        # Actuator Constraint: Solar Radiation Pressure (SRP) limit
        # We assume low-thrust capability (~ milli-Newtons)
        MAX_THRUST = 0.05 
        norm = np.linalg.norm(u_total)
        if norm > MAX_THRUST:
            u_total = u_total / norm * MAX_THRUST
            
        return u_total

# --- SIMULATION LOOP ---
def run_simulation():
    # 1. Setup Scenario: "Hovering" at Unstable L1
    # Target: Stationary at L1 (Physically unnatural, high difficulty)
    target_state = [L1_POS, 0, 0, 0, 0, 0] 
    
    # Initial Condition: Perturbed from equilibrium
    initial_state = [L1_POS + 0.001, 0.001, 0.0, 0.0, 0.0, 0.0]
    
    agent = IENAgent(target_state)
    
    t_span = (0, 50) # Dimensionless time
    dt = 0.01
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    trajectory = []
    entropic_cost = 0.0 # Cumulative Control Effort (integral |u| dt)
    
    state = initial_state
    
    print("🚀 Initiating CR3BP Active Inference Sequence...")
    
    for t in t_eval:
        # A. PERCEPTION & CONTROL
        u = agent.get_control(state)
        
        # Metric: Entropic Cost (Effort), NOT Chemical Fuel
        # Represents momentum transfer required from SRP field.
        entropic_cost += np.linalg.norm(u) * dt
        
        # B. DYNAMICS
        sol = solve_ivp(
            fun=lambda t, y: equations_of_motion(t, y, u),
            t_span=(t, t+dt),
            y0=state,
            rtol=1e-9, atol=1e-9
        )
        state = sol.y[:, -1]
        trajectory.append(state)
        
    trajectory = np.array(trajectory)
    
    return t_eval, trajectory, entropic_cost

# --- VISUALIZATION ---
def plot_cr3bp(trajectory):
    plt.figure(figsize=(10, 8))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=1.5, label='IEN Flight Path')
    plt.plot(L1_POS, 0, 'rx', markersize=10, label='L1 Saddle Point (Target)')
    plt.plot(1-MU, 0, 'ko', markersize=5, label='Moon')
    
    plt.title('IEN Stabilization Test: Forced Equilibrium at L1 (CR3BP)', fontsize=14)
    plt.xlabel('x (Rotating Frame - Dimensionless)')
    plt.ylabel('y (Rotating Frame - Dimensionless)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    
    filename = "CR3BP_Stabilization.png"
    plt.savefig(filename, dpi=300)
    print(f"✅ Simulation Complete. Graph saved as: {filename}")
    plt.show()

if __name__ == "__main__":
    t, traj, cost = run_simulation()
    print(f"📊 Final Thermodynamic Cost (Control Effort): {cost:.6f}")
    plot_cr3bp(traj)