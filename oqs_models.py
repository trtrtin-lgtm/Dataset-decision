import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

def get_hamiltonian(d):
    return np.array([[0.0, d, 0.0],
                     [d, 0.0, d],
                     [0.0, d, 0.0]], dtype=np.complex128)

def get_lindblad_ops_oqs3():
    L1 = np.zeros((3, 3), dtype=np.complex128)
    L1[0, 1] = 1.0
    L2 = np.zeros((3, 3), dtype=np.complex128)
    L2[2, 1] = 1.0
    return [L1, L2]

def get_lindblad_ops_oqs5():
    L1 = np.zeros((3, 3), dtype=np.complex128)
    L1[0, 1] = 1.0
    L2 = np.zeros((3, 3), dtype=np.complex128)
    L2[2, 1] = 1.0
    L3 = np.zeros((3, 3), dtype=np.complex128)
    L3[1, 0] = 1.0
    L4 = np.zeros((3, 3), dtype=np.complex128)
    L4[1, 2] = 1.0
    return [L1, L2, L3, L4]

def build_liouvillian_oqs3(d, gamma1, gamma2):
    H = get_hamiltonian(d)
    I = np.eye(3, dtype=np.complex128)
    L_coh = -1j * (np.kron(I, H) - np.kron(H.T, I))
    L_diss = np.zeros((9, 9), dtype=np.complex128)
    ops = get_lindblad_ops_oqs3()
    rates = [gamma1, gamma2]
    for L, g in zip(ops, rates):
        L_dag = L.conj().T
        term1 = np.kron(L.conj(), L)
        term2 = 0.5 * np.kron(I, L_dag @ L) + 0.5 * np.kron((L_dag @ L).T, I)
        L_diss += g * (term1 - term2)
    return L_coh + L_diss

def lindblad_rhs(t, rho_flat, H, L_ops, rates, timescale=1.0):
    rho = rho_flat.reshape(3, 3)
    d_rho = -1j * (H @ rho - rho @ H)
    for L, g in zip(L_ops, rates):
        L_dag = L.conj().T
        term1 = L @ rho @ L_dag
        term2 = 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L)
        d_rho += g * (term1 - term2)
    return (timescale * d_rho).reshape(9)

def solve_oqs3(d, gamma1, gamma2, t_eval, rho0=None):
    if rho0 is None:
        rho0 = np.zeros((3, 3), dtype=np.complex128)
        rho0[1, 1] = 1.0
    H = get_hamiltonian(d)
    L_ops = get_lindblad_ops_oqs3()
    rates = [gamma1, gamma2]
    sol = solve_ivp(
        lambda t, y: lindblad_rhs(t, y, H, L_ops, rates),
        [0.0, float(t_eval[-1]) + 0.05],
        rho0.reshape(9),
        t_eval=t_eval,
        method="RK45"
    )
    if sol.y.shape[1] != len(t_eval):
        return np.zeros(len(t_eval)), np.zeros(len(t_eval))
    rho_t = sol.y.reshape(3, 3, -1)
    p1 = np.real(rho_t[0, 0, :])
    p2 = np.real(rho_t[2, 2, :])
    return p1, p2

def solve_oqs5(d, gamma1, gamma2, gamma3, gamma4, t_eval, rho0=None):
    if rho0 is None:
        rho0 = np.zeros((3, 3), dtype=np.complex128)
        rho0[1, 1] = 1.0
    H = get_hamiltonian(d)
    L_ops = get_lindblad_ops_oqs5()
    rates = [gamma1, gamma2, gamma3, gamma4]
    sol = solve_ivp(
        lambda t, y: lindblad_rhs(t, y, H, L_ops, rates),
        [0.0, float(t_eval[-1]) + 0.05],
        rho0.reshape(9),
        t_eval=t_eval,
        method="RK45"
    )
    if sol.y.shape[1] != len(t_eval):
        return np.zeros(len(t_eval)), np.zeros(len(t_eval))
    rho_t = sol.y.reshape(3, 3, -1)
    p1 = np.real(rho_t[0, 0, :])
    p2 = np.real(rho_t[2, 2, :])
    return p1, p2

def solve_hybrid(d, gamma1, gamma2, w, t_eval, rho0=None):
    d_eff = (1.0 - w) * d
    g1_eff = w * gamma1
    g2_eff = w * gamma2
    return solve_oqs3(d_eff, g1_eff, g2_eff, t_eval, rho0)
