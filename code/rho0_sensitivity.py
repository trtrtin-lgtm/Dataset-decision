# -*- coding: utf-8 -*-
"""
Phân tích độ nhạy đối với điều kiện ban đầu rho(0)  —  BẢN ĐÃ SỬA.

Dùng ĐÚNG quy ước của pipeline:
  * chỉ các bin có Bin_ValidRatio >= 0.30 (trung vị 9 trong 15 bin)
  * mốc thời gian lấy từ cột t_bin_center_sec (bắt đầu 0.1 s, không phải 0)
  * cùng số tham số: CTMC k=4, OQS4 k=3, Hybrid k=4

Nghiệm tính bằng phân rã riêng của toán tử sinh (linear ODE), nhanh hơn tích phân số,
và được kiểm chứng khớp với solver của pipeline trước khi dùng.

Chạy:  python GUI_App/rho0_sensitivity.py
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN  = os.path.join(ROOT, 'results', 'OQS_Run_20260923_211503_FIXED', 'Data_Excel', 'OQS_Results_Auto.xlsx')
OUTC = os.path.join(ROOT, 'results', 'rho0_sensitivity.csv')
MIN_VR = 0.30

I3 = np.eye(3, dtype=complex)
L12 = np.zeros((3, 3), complex); L12[0, 1] = 1.0   # |1><2|
L32 = np.zeros((3, 3), complex); L32[2, 1] = 1.0   # |3><2|
JUMPS = [L12, L32]

def H_of(d):
    return np.array([[0, d, 0], [d, 0, d], [0, d, 0]], dtype=complex)

def lindblad_super(d, gammas, wu=1.0, wd=1.0):
    """vec quy ước column-major: vec(AXB) = (B^T kron A) vec(X)."""
    H = H_of(d)
    S = -1j * wu * (np.kron(I3, H) - np.kron(H.T, I3))
    for L, g in zip(JUMPS, gammas):
        Ld = L.conj().T; LdL = Ld @ L
        S += wd * g * (np.kron(L.conj(), L) - 0.5 * (np.kron(I3, LdL) + np.kron(LdL.T, I3)))
    return S

def evolve(S, v0, t):
    """v(t_i) = expm(S t_i) v0 cho mọi t_i, qua phân rã riêng (1 lần cho cả vector t)."""
    lam, V = np.linalg.eig(S)
    c = np.linalg.solve(V, v0)
    return (V @ (np.exp(np.outer(lam, t)) * c[:, None])).T      # (len(t), dim)

def pred_oqs(p, rho0, t, hybrid=False):
    if hybrid:
        d, g1, g2, w = p
        S = lindblad_super(d, (g1, g2), wu=(1.0 - w), wd=w)
    else:
        d, g1, g2 = p
        S = lindblad_super(d, (g1, g2))
    R = evolve(S, rho0.reshape(9, order='F'), t)
    R = np.stack([v.reshape(3, 3, order='F') for v in R])
    return np.real(R[:, 0, 0]), np.real(R[:, 2, 2])

def pred_ctmc(p, P0, t):
    q12, q21, q23, q32 = p
    Q = np.array([[-q12, q12, 0.0], [q21, -(q21 + q23), q23], [0.0, q32, -q32]], dtype=complex)
    V = evolve(Q.T, P0.astype(complex), t)      # dP/dt = P Q  ->  d(P^T)/dt = Q^T P^T
    return np.real(V[:, 0]), np.real(V[:, 2])

BOUNDS = {'ctmc': [(0, 5)] * 4, 'oqs': [(0, 5)] * 3, 'hybrid': [(0, 5)] * 3 + [(0.0, 1.0)]}
KPAR   = {'ctmc': 4, 'oqs': 3, 'hybrid': 4}

def fit(kind, y1, y2, rho0, t, nstart, rng):
    P0 = np.real(np.diag(rho0)).copy()
    best, bx = np.inf, None
    for _ in range(nstart):
        x0 = rng.uniform(0.05, 2.0, len(BOUNDS[kind]))
        if kind == 'hybrid':
            x0[3] = rng.uniform(0.05, 0.95)
        def obj(p):
            try:
                a, b = pred_ctmc(p, P0, t) if kind == 'ctmc' else pred_oqs(p, rho0, t, kind == 'hybrid')
            except Exception:
                return 1e6
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                return 1e6
            return float(np.sum((y1 - a) ** 2) + np.sum((y2 - b) ** 2))
        try:
            r = minimize(obj, x0, bounds=BOUNDS[kind], method='L-BFGS-B')
            if r.fun < best:
                best, bx = float(r.fun), r.x
        except Exception:
            pass
    return best, bx

def validate():
    """Kiểm chứng solver của script này khớp với solver của pipeline."""
    import re
    sys.path.insert(0, os.path.join(ROOT, 'GUI_App'))
    from modules.data_processor import solve_oqs_model
    from modules.classical_model import solve_classical_model
    fit_df = pd.read_excel(RUN, sheet_name='OQS_Fit_Results')
    pr = pd.read_excel(RUN, sheet_name='Prob_5bins'); pr = pr[pr.Trial_OK == 1]
    pk = lambda s: np.array([float(x) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', str(s))])
    r = fit_df.iloc[0]
    g = pr[(pr.subject == r.subject) & (pr.TrialID == r.TrialID)].sort_values('bin')
    m = (g.Bin_ValidRatio >= MIN_VR).to_numpy()
    t = g.t_bin_center_sec.to_numpy(float)[m]
    rho0 = np.zeros((3, 3), complex); rho0[1, 1] = 1.0
    a1, b1 = solve_oqs_model(pk(r.Params_OQS4), t, 'OQS4')
    a2, b2 = pred_oqs(pk(r.Params_OQS4)[:3], rho0, t)
    e_oqs = max(np.max(np.abs(a1 - a2)), np.max(np.abs(b1 - b2)))
    c1, d1 = solve_classical_model(pk(r.Params_CTMC), t)
    c2, d2 = pred_ctmc(pk(r.Params_CTMC)[:4], np.array([0., 1., 0.]), t)
    e_ctmc = max(np.max(np.abs(c1 - c2)), np.max(np.abs(d1 - d2)))
    print('  kiểm chứng solver:  max|lệch| OQS = %.2e   CTMC = %.2e' % (e_oqs, e_ctmc))
    assert e_oqs < 1e-4 and e_ctmc < 1e-4, 'solver KHÔNG khớp pipeline'
    print('  -> khớp pipeline, tiếp tục.\n')

def main(nstart=12):
    print('Kiểm chứng trước khi chạy...')
    validate()
    pr = pd.read_excel(RUN, sheet_name='Prob_5bins'); pr = pr[pr.Trial_OK == 1]
    rng = np.random.default_rng(2026)
    rows, groups, t0 = [], list(pr.groupby(['subject', 'TrialID'])), time.time()
    for idx, ((subj, tid), g) in enumerate(groups, 1):
        g = g.sort_values('bin')
        m = (g.Bin_ValidRatio >= MIN_VR).to_numpy()
        if m.sum() < 3:
            continue
        t  = g.t_bin_center_sec.to_numpy(float)[m]
        y1 = g.Prob_Op1.to_numpy(float)[m]
        y2 = g.Prob_Op2.to_numpy(float)[m]

        rho_fixed = np.zeros((3, 3), complex); rho_fixed[1, 1] = 1.0
        diag = np.array([y1[0], max(0.0, 1.0 - y1[0] - y2[0]), y2[0]], float)
        if diag.sum() <= 1e-9:
            continue
        rho_data = np.diag(diag / diag.sum()).astype(complex)

        rec = {'subject': subj, 'TrialID': tid, 'n_fit_bins': int(m.sum()), 'n_all_bins': len(g)}
        for lab, r0 in (('fixed', rho_fixed), ('data', rho_data)):
            for kind in ('ctmc', 'oqs', 'hybrid'):
                sse, x = fit(kind, y1, y2, r0, t, nstart, rng)
                rec['SSE_%s_%s' % (kind, lab)] = sse
                if kind == 'hybrid' and x is not None:
                    rec['w_%s' % lab] = float(x[3])
        rows.append(rec)
        if idx % 10 == 0 or idx == len(groups):
            pd.DataFrame(rows).to_csv(OUTC, index=False)
            el = time.time() - t0
            print('  %3d/%d  |  %.0fs  |  còn ~%.0fs' % (idx, len(groups), el, el / idx * (len(groups) - idx)), flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUTC, index=False)
    print('\n' + '=' * 74)
    print('ĐỘ NHẠY rho(0)   —   n = %d lượt thử,  trung vị %d/%d bin dùng để khớp'
          % (len(df), df.n_fit_bins.median(), df.n_all_bins.median()))
    print('=' * 74)
    nobs = 2 * df.n_fit_bins.to_numpy(float)
    summary = {}
    for lab, title in (('fixed', 'rho(0) = |2><2|   (như pipeline hiện tại)'),
                       ('data',  'rho(0) ước lượng từ bin đầu tiên')):
        print('\n--- %s ---' % title)
        sse = {k: df['SSE_%s_%s' % (k, lab)].to_numpy(float) for k in KPAR}
        for k in KPAR:
            print('    %-7s SSE trung bình %7.4f   trung vị %7.4f' % (k.upper(), sse[k].mean(), np.median(sse[k])))
        B = np.vstack([nobs * np.log(np.maximum(sse[k], 1e-12) / nobs) + KPAR[k] * np.log(nobs) for k in KPAR]).T
        win = np.array(list(KPAR))[B.argmin(1)]
        counts = {k: int((win == k).sum()) for k in KPAR}
        summary[lab] = counts
        print('    BIC thắng:  ' + '   '.join('%s %d' % (k.upper(), counts[k]) for k in KPAR))
        if 'w_%s' % lab in df:
            print('    w: trung vị %.3f  IQR [%.3f, %.3f]'
                  % (df['w_%s' % lab].median(), df['w_%s' % lab].quantile(.25), df['w_%s' % lab].quantile(.75)))
    print('\n>>> Thứ hạng có đổi khi thay rho(0) không? ',
          'KHÔNG' if max(summary['fixed'], key=summary['fixed'].get) == max(summary['data'], key=summary['data'].get) else 'CÓ')
    print('Đã ghi:', OUTC)

if __name__ == '__main__':
    main()
