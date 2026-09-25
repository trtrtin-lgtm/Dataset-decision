import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. Read Data
f = 'Calculate/OQS_Results_Auto.xlsx'
if not os.path.exists(f):
    print("Khong tim thay OQS_Results_Auto.xlsx. Vui long chay tinh toan truoc.")
    sys.exit(0)

df_fit = pd.read_excel(f, sheet_name='OQS_Fit_Results')
df_trans = pd.read_excel(f, sheet_name='Transitions_trial_5bins')

def extract_w(param_str):
    if pd.isna(param_str): return np.nan
    param_str = str(param_str).strip('[]')
    parts = param_str.replace('\n', ' ').replace(',', ' ').split()
    if len(parts) >= 4: return float(parts[3])
    return np.nan

df_fit['w'] = df_fit['Params_Hybrid'].apply(extract_w)
df_fit = df_fit.dropna(subset=['w'])

df_trans['total_saccades'] = df_trans['b12'] + df_trans['b21']
df_trans['total_measurements'] = df_trans['b12'] + df_trans['b21'] + df_trans['w1'] + df_trans['w2']

trial_trans = df_trans.groupby(['subject', 'TrialID']).agg({
    'total_saccades': 'sum',
    'total_measurements': 'sum'
}).reset_index()

df_merged = pd.merge(df_fit, trial_trans, on=['subject', 'TrialID'], how='inner')

if len(df_merged) < 3:
    print("Khong du du lieu de chay tuong quan.")
    sys.exit(0)

# 5. Correlation
r_meas, p_meas = stats.pearsonr(df_merged['total_measurements'], df_merged['w'])
r_sacc, p_sacc = stats.pearsonr(df_merged['total_saccades'], df_merged['w'])

print("--- PHAN TICH DOT PHA: COGNITIVE DECOHERENCE ---")
print(f"Tong so trials hop le: {len(df_merged)}")
print(f"Tuong quan Dao mat (Saccades) va w: r = {r_sacc:.4f}, p = {p_sacc:.4f}")
if p_sacc < 0.1:
    print("=> Dao mat cang nhieu, tinh luong tu cang manh (w giam).")

fig_dir = r'../Figures'
os.makedirs(fig_dir, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.2)
fig, ax = plt.subplots(figsize=(8, 6))

sns.regplot(
    data=df_merged, x='total_saccades', y='w', 
    scatter_kws={'alpha':0.6, 's':80, 'color':'teal'}, 
    line_kws={'color':'darkred', 'lw':2},
    ax=ax
)

ax.set_title('Cognitive Decoherence:\nVisual Saccades vs. Quantum Collapse', fontweight='bold')
ax.set_xlabel('Total Visual Saccades (Transitions in 3s)')
ax.set_ylabel('Hybrid Weight w\n(0 = Pure Quantum, 1 = Pure Classical)')
ax.set_ylim(-0.05, 1.05)

textstr = f'N = {len(df_merged)} trials\nPearson r = {r_sacc:.3f}\np-value = {p_sacc:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.tight_layout()
out_plot = os.path.join(fig_dir, 'Fig14_Cognitive_Decoherence.png')
plt.savefig(out_plot, dpi=300)
print(f"Da luu bieu do tai: {out_plot}")
df_merged[['subject', 'TrialID', 'total_saccades', 'total_measurements', 'w']].to_csv('Calculate/Decoherence_Data.csv', index=False)
