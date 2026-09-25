import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import os
import sys

# Import solvers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.data_processor import solve_oqs_model
from modules.classical_model import solve_classical_model

sys.stdout.reconfigure(encoding='utf-8')

print("Reading results...")
f = r'Calculate/OQS_Results_Auto.xlsx'
df_fit = pd.read_excel(f, sheet_name='OQS_Fit_Results')
df_prob = pd.read_excel(f, sheet_name='Prob_5bins')

w_values = []
hybrid_sses = []

def parse_params(param_str):
    if pd.isna(param_str): return None
    param_str = param_str.strip('[]')
    # handle multiline string or commas if any
    param_str = param_str.replace('\n', ' ').replace(',', ' ')
    return np.array([float(x) for x in param_str.split()])

print("Fitting Hybrid w-parameter...")
for idx, row in df_fit.iterrows():
    subj = row['subject']
    tid = row['TrialID']
    
    params_oqs4 = parse_params(row['Params_OQS4'])
    params_ctmc = parse_params(row['Params_CTMC'])
    
    if params_oqs4 is None or params_ctmc is None:
        continue
        
    prob_rows = df_prob[(df_prob['subject'] == subj) & (df_prob['TrialID'] == tid) & (df_prob['ValidBins'] >= 3) & (df_prob['Bin_ValidRatio'] >= 0.3)]
    if len(prob_rows) < 3:
        continue
        
    prob_rows = prob_rows.sort_values('t_bin_center_sec')
    t_eval = prob_rows['t_bin_center_sec'].values
    p11_true = prob_rows['Prob_Op1'].values
    p33_true = prob_rows['Prob_Op2'].values
    
    try:
        p11_oqs, p33_oqs = solve_oqs_model(params_oqs4, t_eval, 'OQS4')
        p11_ctmc, p33_ctmc = solve_classical_model(params_ctmc, t_eval)
        
        # Define objective function for w
        def obj_w(w):
            p11_hyb = w * p11_ctmc + (1 - w) * p11_oqs
            p33_hyb = w * p33_ctmc + (1 - w) * p33_oqs
            sse = np.sum((p11_true - p11_hyb)**2) + np.sum((p33_true - p33_hyb)**2)
            return sse
            
        res = minimize_scalar(obj_w, bounds=(0, 1), method='bounded')
        w_opt = res.x
        w_values.append(w_opt)
        hybrid_sses.append(res.fun)
    except Exception as e:
        print(f"Error for {subj} {tid}: {e}")

w_arr = np.array(w_values)
print(f"Total valid hybrid fits: {len(w_arr)}")
print(f"Mean w = {np.mean(w_arr):.4f}")
print(f"Median w = {np.median(w_arr):.4f}")
print(f"w < 0.1 (Quantum): {np.sum(w_arr < 0.1)}")
print(f"w > 0.9 (Classical): {np.sum(w_arr > 0.9)}")
print(f"0.1 <= w <= 0.9 (Hybrid): {np.sum((w_arr >= 0.1) & (w_arr <= 0.9))}")

# Plot
plt.figure(figsize=(8,6))
plt.hist(w_arr, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(w_arr), color='red', linestyle='dashed', linewidth=2, label=f'Mean w = {np.mean(w_arr):.2f}')
plt.title('Distribution of Hybrid Parameter w\n(0 = Quantum, 1 = Classical CTMC)')
plt.xlabel('Hybrid Weight w')
plt.ylabel('Frequency (Trials)')
plt.legend()
plt.tight_layout()

fig_dir = r'../Figures'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
plt.savefig(os.path.join(fig_dir, 'Fig12_HybridWeightDistribution.png'), dpi=300)
print("Plot saved to Fig12_HybridWeightDistribution.png")

df_w = pd.DataFrame({'w': w_arr, 'SSE_Hybrid': hybrid_sses})
df_w.to_csv('Calculate/Hybrid_W_Results.csv', index=False)
