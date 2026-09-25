import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def get_hamiltonian(d: float) -> np.ndarray:
    return np.array([[0.0, d, 0.0],
                     [d, 0.0, d],
                     [0.0, d, 0.0]], dtype=np.complex128)

def get_lindblad_ops_oqs4():
    L12 = np.zeros((3, 3), dtype=np.complex128); L12[0, 1] = 1.0
    L32 = np.zeros((3, 3), dtype=np.complex128); L32[2, 1] = 1.0
    return [L12, L32]

def get_lindblad_ops_oqs6():
    ops = get_lindblad_ops_oqs4()
    L21 = np.zeros((3, 3), dtype=np.complex128); L21[1, 0] = 1.0
    L23 = np.zeros((3, 3), dtype=np.complex128); L23[1, 2] = 1.0
    return ops + [L21, L23]

def lindblad_rhs(t, rho_flat, H, L_ops, rates, timescale):
    rho = rho_flat.reshape(3, 3)
    d_rho = -1j * (H @ rho - rho @ H)
    for L, gamma in zip(L_ops, rates):
        L_dag = L.conj().T
        term1 = L @ rho @ L_dag
        LdagL = L_dag @ L
        term2 = LdagL @ rho + rho @ LdagL
        d_rho += gamma * (term1 - 0.5 * term2)
    return (timescale * d_rho).reshape(9)

def solve_oqs_model(params, t_eval, model_type="OQS4"):
    d = float(params[0])
    timescale = float(params[-1])

    if model_type == "OQS4":
        rates = [float(params[1]), float(params[2])]
        L_ops = get_lindblad_ops_oqs4()
    else:
        rates = [float(params[1]), float(params[2]), float(params[3]), float(params[4])]
        L_ops = get_lindblad_ops_oqs6()

    H = get_hamiltonian(d)
    rho0 = np.zeros((3, 3), dtype=np.complex128)
    rho0[1, 1] = 1.0
    y0 = rho0.reshape(9)

    def system_wrapper(tt, yy):
        return lindblad_rhs(tt, yy, H, L_ops, rates, timescale)

    sol = solve_ivp(
        system_wrapper,
        [0.0, float(t_eval[-1]) + 0.1],
        y0,
        t_eval=t_eval,
        method="RK45"
    )
    if sol.y.shape[1] != len(t_eval):
        return np.zeros(len(t_eval)), np.zeros(len(t_eval))

    rho_t = sol.y.reshape(3, 3, -1)
    p11 = np.real(rho_t[0, 0, :])
    p33 = np.real(rho_t[2, 2, :])
    return p11, p33

def objective_function_oqs(params, t_eval, p11_true, p33_true, model_type):
    try:
        p11_pred, p33_pred = solve_oqs_model(params, t_eval, model_type)
        if len(p11_pred) != len(p11_true):
            return 1e6
        return float(np.sum((p11_true - p11_pred) ** 2) + np.sum((p33_true - p33_pred) ** 2))
    except Exception:
        return 1e6
