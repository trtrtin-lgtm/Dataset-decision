import numpy as np
from scipy.optimize import minimize
from oqs_models import solve_oqs3
from classical_models import solve_ctmc

def calc_sse(y_true, y_pred):
    return float(np.sum((y_true - y_pred) ** 2))

def calc_pseudo_bic(sse, n, k):
    return float(n * np.log(max(sse, 1e-12) / n) + k * np.log(n))

def calc_aic(sse, n, k):
    return float(n * np.log(max(sse, 1e-12) / n) + 2 * k)

def calc_bartlett_neff(n, r1):
    return float(n * (1.0 - r1) / (1.0 + r1))

def fit_oqs3(p1_true, p2_true, t_eval, n_starts=20, bound=5.0):
    best_res = None
    best_loss = 1e12
    bounds = [(0.0, bound), (0.0, bound), (0.0, bound)]
    def obj(params):
        d, g1, g2 = params
        p1, p2 = solve_oqs3(d, g1, g2, t_eval)
        return calc_sse(p1_true, p1) + calc_sse(p2_true, p2)
    starts = [np.random.uniform(0.0, bound, size=3) for _ in range(n_starts)]
    for x0 in starts:
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
        if res.fun < best_loss:
            best_loss = res.fun
            best_res = res
    n_obs = 2 * len(t_eval)
    bic = calc_pseudo_bic(best_loss, n_obs, k=3)
    aic = calc_aic(best_loss, n_obs, k=3)
    return {"params": best_res.x, "sse": best_loss, "bic": bic, "aic": aic}

def fit_ctmc4(p1_true, p2_true, t_eval, n_starts=20, bound=5.0):
    best_res = None
    best_loss = 1e12
    bounds = [(0.0, bound), (0.0, bound), (0.0, bound), (0.0, bound)]
    def obj(params):
        p1, p2 = solve_ctmc(params, t_eval, model_type="CTMC4")
        return calc_sse(p1_true, p1) + calc_sse(p2_true, p2)
    starts = [np.random.uniform(0.0, bound, size=4) for _ in range(n_starts)]
    for x0 in starts:
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
        if res.fun < best_loss:
            best_loss = res.fun
            best_res = res
    n_obs = 2 * len(t_eval)
    bic = calc_pseudo_bic(best_loss, n_obs, k=4)
    aic = calc_aic(best_loss, n_obs, k=4)
    return {"params": best_res.x, "sse": best_loss, "bic": bic, "aic": aic}

def fit_ctmc2a(p1_true, p2_true, t_eval, n_starts=20, bound=5.0):
    best_res = None
    best_loss = 1e12
    bounds = [(0.0, bound), (0.0, bound)]
    def obj(params):
        p1, p2 = solve_ctmc(params, t_eval, model_type="CTMC2_A")
        return calc_sse(p1_true, p1) + calc_sse(p2_true, p2)
    starts = [np.random.uniform(0.0, bound, size=2) for _ in range(n_starts)]
    for x0 in starts:
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
        if res.fun < best_loss:
            best_loss = res.fun
            best_res = res
    n_obs = 2 * len(t_eval)
    bic = calc_pseudo_bic(best_loss, n_obs, k=2)
    aic = calc_aic(best_loss, n_obs, k=2)
    return {"params": best_res.x, "sse": best_loss, "bic": bic, "aic": aic}

def temporal_cross_validation(p1_true, p2_true, t_eval, model="oqs3", method="interleaved"):
    n = len(t_eval)
    if method == "interleaved":
        train_idx = np.arange(0, n, 2)
        test_idx = np.arange(1, n, 2)
    elif method == "forward":
        half = n // 2
        train_idx = np.arange(0, half)
        test_idx = np.arange(half, n)
    else:
        raise ValueError(f"Unknown validation method: {method}")
    if model == "oqs3":
        fit = fit_oqs3(p1_true[train_idx], p2_true[train_idx], t_eval[train_idx], n_starts=10)
        p1_pred, p2_pred = solve_oqs3(*fit["params"], t_eval[test_idx])
    elif model == "ctmc4":
        fit = fit_ctmc4(p1_true[train_idx], p2_true[train_idx], t_eval[train_idx], n_starts=10)
        p1_pred, p2_pred = solve_ctmc(fit["params"], t_eval[test_idx], model_type="CTMC4")
    else:
        raise ValueError(f"Unknown model: {model}")
    held_out_sse = calc_sse(p1_true[test_idx], p1_pred) + calc_sse(p2_true[test_idx], p2_pred)
    return held_out_sse
