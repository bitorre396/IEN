import numpy as np

# --- PHYSICAL PARAMETERS ---
SRP_PRESSURE = 4.5e-6   # N/m^2 at 1 AU
AREA = 20.0             # m^2
MASS = 500.0            # kg
GAIN = 1.0              # Control Gain

def KL_gaussian_closed_form(mu_p, Sigma_p, mu_q, Sigma_q):
    """
    Computes closed-form KL Divergence D_KL(P || Q) 
    between two multivariate Gaussians.
    P: Target Model (Mission Plan)
    Q: Observer Belief (Estimated State)
    """
    k = len(mu_p)
    
    # Invert Q covariance (with stability check)
    try:
        inv_Sigma_q = np.linalg.inv(Sigma_q)
    except np.linalg.LinAlgError:
        inv_Sigma_q = np.linalg.pinv(Sigma_q)

    # 1. Trace Term
    trace_term = np.trace(inv_Sigma_q @ Sigma_p)
    
    # 2. Quadratic (Mahalanobis) Term
    diff = mu_q - mu_p
    quad_term = diff.T @ inv_Sigma_q @ diff
    
    # 3. Log-Determinant Term
    (_, logdet_q) = np.linalg.slogdet(Sigma_q)
    (_, logdet_p) = np.linalg.slogdet(Sigma_p)
    log_det_term = logdet_q - logdet_p
    
    # D_KL formula
    return 0.5 * (trace_term + quad_term - k + log_det_term)

def compute_gradient(observer, P_target, current_action):
    """
    Estimates the gradient of Entropy w.r.t Action (dKL/da).
    Uses finite differences to simulate the effect of 
    changing panel orientation on future belief state.
    """
    delta = 1e-4 # Small perturbation (radians)
    
    # Forward simulation step
    Q_next_plus = observer.predict(current_action + delta)
    Q_next_curr = observer.predict(current_action)
    
    # Calculate Divergence for both futures
    kl_plus = KL_gaussian_closed_form(
        P_target['mu'], P_target['Sigma'],
        Q_next_plus['mu'], Q_next_plus['Sigma']
    )
    
    kl_curr = KL_gaussian_closed_form(
        P_target['mu'], P_target['Sigma'],
        Q_next_curr['mu'], Q_next_curr['Sigma']
    )
    
    # dKL / da
    return (kl_plus - kl_curr) / delta

def control_law(gradient):
    """
    Applies Active Inference Control.
    Action moves against the gradient of entropy.
    """
    return -GAIN * gradient
