import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def ctmc_rhs(t, P, Q):
    # P is a probability vector [P_Op1, P_Mid, P_Op2]
    # dP/dt = P @ Q
    return P @ Q

def solve_classical_model(params, t_eval):
    # params = [q12, q21, q23, q32]
    q12, q21, q23, q32 = [float(p) for p in params]
    timescale = 1.0
    
    Q = np.array([
        [-q12, q12, 0.0],
        [q21, -(q21 + q23), q23],
        [0.0, q32, -q32]
    ], dtype=np.float64)
    
    Q = Q * timescale
    
    P0 = np.array([0.0, 1.0, 0.0], dtype=np.float64) # Start in Mid
    
    sol = solve_ivp(
        lambda t, y: ctmc_rhs(t, y, Q),
        [0.0, float(t_eval[-1]) + 0.1],
        P0,
        t_eval=t_eval,
        method="RK45"
    )
    
    if sol.y.shape[1] != len(t_eval):
        return np.zeros(len(t_eval)), np.zeros(len(t_eval))
        
    P_t = sol.y
    P_Op1 = P_t[0, :]
    P_Op2 = P_t[2, :]
    
    return P_Op1, P_Op2

def objective_function_classical(params, t_eval, p1_true, p3_true):
    try:
        p1_pred, p3_pred = solve_classical_model(params, t_eval)
        if len(p1_pred) != len(p1_true):
            return 1e6
        return float(np.sum((p1_true - p1_pred) ** 2) + np.sum((p3_true - p3_pred) ** 2))
    except Exception:
        return 1e6
