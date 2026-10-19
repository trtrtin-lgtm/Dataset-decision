import numpy as np
from scipy.integrate import solve_ivp

def build_ctmc_generator(rates, model_type="CTMC4"):
    if model_type == "CTMC4":
        q12, q21, q23, q32 = rates
        Q = np.array([
            [-q12, q12, 0.0],
            [q21, -(q21 + q23), q23],
            [0.0, q32, -q32]
        ], dtype=np.float64)
    elif model_type == "CTMC3_R":
        q21, q23, r = rates
        Q = np.array([
            [-r, r, 0.0],
            [q21, -(q21 + q23), q23],
            [0.0, r, -r]
        ], dtype=np.float64)
    elif model_type == "CTMC3_E":
        q12, q32, e = rates
        Q = np.array([
            [-q12, q12, 0.0],
            [e, -2.0 * e, e],
            [0.0, q32, -q32]
        ], dtype=np.float64)
    elif model_type == "CTMC2_A":
        q21, q23 = rates
        Q = np.array([
            [0.0, 0.0, 0.0],
            [q21, -(q21 + q23), q23],
            [0.0, 0.0, 0.0]
        ], dtype=np.float64)
    elif model_type == "CTMC2_S":
        e, r = rates
        Q = np.array([
            [-r, r, 0.0],
            [e, -2.0 * e, e],
            [0.0, r, -r]
        ], dtype=np.float64)
    else:
        raise ValueError(f"Unknown classical model type: {model_type}")
    return Q

def ctmc_rhs(t, P, Q):
    return P @ Q

def solve_ctmc(rates, t_eval, model_type="CTMC4", P0=None):
    if P0 is None:
        P0 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    Q = build_ctmc_generator(rates, model_type)
    sol = solve_ivp(
        lambda t, y: ctmc_rhs(t, y, Q),
        [0.0, float(t_eval[-1]) + 0.05],
        P0,
        t_eval=t_eval,
        method="RK45"
    )
    if sol.y.shape[1] != len(t_eval):
        return np.zeros(len(t_eval)), np.zeros(len(t_eval))
    p1 = sol.y[0, :]
    p2 = sol.y[2, :]
    return p1, p2
