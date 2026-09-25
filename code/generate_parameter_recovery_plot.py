import os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import pearsonr
from modules.data_processor import solve_oqs_model, objective_function

np.random.seed(42)
N_SIM = 100
t_eval = np.linspace(0, 3.0, 15)

true_d = np.random.uniform(0.5, 2.5, N_SIM)
true_g1 = np.random.uniform(0.2, 2.5, N_SIM)
true_g2 = np.random.uniform(0.2, 2.5, N_SIM)

est_d = []
est_g1 = []
est_g2 = []

b_d = (0.01, 5.0); b_a = (1e-6, 5.0)
bounds = [b_d, b_a, b_a]

sigma_noise = 0.316

for i in range(N_SIM):
    p11, p33 = solve_oqs_model([true_d[i], true_g1[i], true_g2[i]], t_eval, 'OQS4')
    p11_n = np.clip(p11 + np.random.normal(0, sigma_noise, len(p11)), 0, 1)
    p33_n = np.clip(p33 + np.random.normal(0, sigma_noise, len(p33)), 0, 1)
    
    best_res = None
    for _ in range(5):
        x0 = [np.random.uniform(*b_d), np.random.uniform(*b_a), np.random.uniform(*b_a)]
        res = minimize(objective_function, x0, args=(t_eval, p11_n, p33_n, 'OQS4'), bounds=bounds, method='L-BFGS-B')
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    est_d.append(best_res.x[0])
    est_g1.append(best_res.x[1])
    est_g2.append(best_res.x[2])

est_d = np.array(est_d)
est_g1 = np.array(est_g1)
est_g2 = np.array(est_g2)

stats = {}
for name, tr, es in [('d', true_d, est_d), ('gamma1', true_g1, est_g1), ('gamma2', true_g2, est_g2)]:
    bias = float(np.mean(es - tr))
    rmse = float(np.sqrt(np.mean((es - tr)**2)))
    r, _ = pearsonr(tr, es)
    bound_hits = float(np.mean((es < 0.02) | (es > 4.98)))
    stats[name] = {'bias': bias, 'rmse': rmse, 'r': float(r), 'bound_hits': bound_hits}

print(stats)

ROOT = r"E:\ET\ĐA Đánh giá sự lưỡng lự bằng mô hình lượng tử OQS4 và OQS6-20260828T140941Z-1-001"
FIGDIR = os.path.join(ROOT, 'Figures', 'paper')
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False})
C1, C2 = '#1f4e79', '#c0504d'

def plot_recovery(lang='VI'):
    suffix = '_VI' if lang == 'VI' else '_EN'
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    
    params = [
        ('d', true_d, est_d, 'rad/s'),
        (r'\gamma_1', true_g1, est_g1, '1/s'),
        (r'\gamma_2', true_g2, est_g2, '1/s')
    ]
    names = ['d', 'gamma1', 'gamma2']
    
    for idx, (p_label, tr, es, unit) in enumerate(params):
        ax = axes[idx]
        ax.scatter(tr, es, color=C1, alpha=0.7, edgecolors='none', s=35)
        
        lims = [0, 3.0]
        ax.plot(lims, lims, color=C2, ls='--', lw=1.5, label='y = x')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        st = stats[names[idx]]
        txt = f"r = {st['r']:.3f}\nRMSE = {st['rmse']:.3f}\nBias = {st['bias']:+.3f}"
        ax.text(0.05, 0.92, txt, transform=ax.transAxes, va='top', ha='left',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#ccc'))
        
        if lang == 'VI':
            ax.set_xlabel(f"Giá trị thực ${p_label}$ ({unit})")
            ax.set_ylabel(f"Ước lượng ${p_label}$ ({unit})")
        else:
            ax.set_xlabel(f"True ${p_label}$ ({unit})")
            ax.set_ylabel(f"Recovered ${p_label}$ ({unit})")
            
    fig.tight_layout()
    out_file = os.path.join(FIGDIR, f'Fig5_ParameterRecovery{suffix}.png')
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

plot_recovery('VI')
plot_recovery('EN')
