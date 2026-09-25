# -*- coding: utf-8 -*-
"""
External Validation Pipeline on the Krajbich et al. (2010) Nature Neuroscience Benchmark Dataset.
Ingests raw fixations, reconstructs continuous gaze dwell trajectories, and fits OQS4 vs CTMC.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from modules.data_processor import solve_oqs_model, objective_function
from modules.classical_model import solve_classical_model, objective_function_classical

def bin_trial_fixations(tr_df, n_bins=10):
    rt = float(tr_df['rt'].iloc[0])
    if rt < 600.0 or len(tr_df) < 2:
        return None
    
    curr = 0.0
    intervals = []
    for _, row in tr_df.iterrows():
        dur = float(row['event_duration'])
        roi = float(row['roi'])
        intervals.append((curr, curr + dur, roi))
        curr += dur
        
    total_time = max(curr, rt)
    bin_edges = np.linspace(0, total_time, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) / 1000.0 # seconds
    
    p1 = np.zeros(n_bins)
    p3 = np.zeros(n_bins)
    
    for i in range(n_bins):
        b_start, b_end = bin_edges[i], bin_edges[i+1]
        b_dur = b_end - b_start
        dur_left = 0.0
        dur_right = 0.0
        for (i_start, i_end, roi) in intervals:
            overlap = max(0.0, min(b_end, i_end) - max(b_start, i_start))
            if overlap > 0:
                if roi == 1.0:
                    dur_left += overlap
                elif roi == 2.0:
                    dur_right += overlap
        p1[i] = min(1.0, dur_left / b_dur)
        p3[i] = min(1.0, dur_right / b_dur)
        
    return bin_centers, p1, p3

def run_krajbich_validation():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dta_path = os.path.join(ROOT, 'Literature', 'data_nature2010.dta')
    
    if not os.path.exists(dta_path):
        raise FileNotFoundError(f"Missing {dta_path}")
        
    print(f"Loading Krajbich 2010 dataset from: {dta_path}")
    df = pd.read_stata(dta_path)
    
    subjects = sorted(df['subject'].unique())
    print(f"Loaded {len(df)} fixations across {len(subjects)} subjects.")
    
    # Bounds
    b_oqs = [(0.01, 5.0), (1e-6, 5.0), (1e-6, 5.0)]
    b_ctmc = [(1e-6, 10.0)] * 4
    
    results = []
    t0 = time.time()
    total_processed = 0
    
    print("Beginning model fitting across subjects...")
    for s_idx, subj in enumerate(subjects):
        df_s = df[df['subject'] == subj]
        trials = sorted(df_s['trial'].unique())
        
        for tr_id in trials:
            tr_df = df_s[df_s['trial'] == tr_id]
            binned = bin_trial_fixations(tr_df, n_bins=10)
            if binned is None:
                continue
                
            t_eval, p1_true, p3_true = binned
            n_obs = len(t_eval) * 2 # 20 observations per trial (p1 and p3 over 10 bins)
            
            # 1. Fit OQS4 (k=3)
            best_oqs = None
            for _ in range(3):
                x0 = [np.random.uniform(*b_oqs[0]), np.random.uniform(*b_oqs[1]), np.random.uniform(*b_oqs[2])]
                res_o = minimize(objective_function, x0, args=(t_eval, p1_true, p3_true, 'OQS4'),
                                 bounds=b_oqs, method='L-BFGS-B')
                if best_oqs is None or res_o.fun < best_oqs.fun:
                    best_oqs = res_o
                    
            # 2. Fit CTMC (k=4)
            best_ctmc = None
            for _ in range(3):
                x0 = [np.random.uniform(*b_ctmc[0]), np.random.uniform(*b_ctmc[1]),
                      np.random.uniform(*b_ctmc[2]), np.random.uniform(*b_ctmc[3])]
                res_c = minimize(objective_function_classical, x0, args=(t_eval, p1_true, p3_true),
                                 bounds=b_ctmc, method='L-BFGS-B')
                if best_ctmc is None or res_c.fun < best_ctmc.fun:
                    best_ctmc = res_c
                    
            sse_oqs = float(best_oqs.fun)
            sse_ctmc = float(best_ctmc.fun)
            
            bic_oqs = n_obs * np.log(max(sse_oqs / n_obs, 1e-12)) + 3 * np.log(n_obs)
            bic_ctmc = n_obs * np.log(max(sse_ctmc / n_obs, 1e-12)) + 4 * np.log(n_obs)
            
            aic_oqs = n_obs * np.log(max(sse_oqs / n_obs, 1e-12)) + 2 * 3
            aic_ctmc = n_obs * np.log(max(sse_ctmc / n_obs, 1e-12)) + 2 * 4
            
            delta_bic = bic_ctmc - bic_oqs # Positive means OQS4 is preferred
            delta_aic = aic_ctmc - aic_oqs
            
            d_val, g1_val, g2_val = best_oqs.x
            bound_hit = int((g1_val < 0.02) or (g1_val > 4.98) or (g2_val < 0.02) or (g2_val > 4.98) or (d_val < 0.02))
            
            results.append({
                'subject': subj,
                'trial': tr_id,
                'rt': float(tr_df['rt'].iloc[0]),
                'choice': float(tr_df['choice'].iloc[0]),
                'num_fixations': len(tr_df),
                'SSE_OQS4': sse_oqs,
                'SSE_CTMC': sse_ctmc,
                'BIC_OQS4': bic_oqs,
                'BIC_CTMC': bic_ctmc,
                'Delta_BIC': delta_bic,
                'AIC_OQS4': aic_oqs,
                'AIC_CTMC': aic_ctmc,
                'Delta_AIC': delta_aic,
                'BestModel_BIC': 'OQS4' if delta_bic > 0 else 'Classical_CTMC',
                'OQS_d': d_val,
                'OQS_g1': g1_val,
                'OQS_g2': g2_val,
                'BoundHit': bound_hit
            })
            total_processed += 1
            
        if (s_idx + 1) % 5 == 0 or (s_idx + 1) == len(subjects):
            el = time.time() - t0
            print(f"  Processed {s_idx + 1}/{len(subjects)} subjects ({total_processed} trials) | {el:.1f}s elapsed")

    df_res = pd.DataFrame(results)
    out_dir = os.path.join(ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'Krajbich_2010_Validation_Results.xlsx')
    df_res.to_excel(out_file, index=False)
    print(f"\nSaved {len(df_res)} trial validation results to: {out_file}")
    
    # Statistical Summary
    n_trials = len(df_res)
    oqs_wins = (df_res['BestModel_BIC'] == 'OQS4').sum()
    ctmc_wins = (df_res['BestModel_BIC'] == 'Classical_CTMC').sum()
    
    delta_bic = df_res['Delta_BIC']
    median_dbic = delta_bic.median()
    iqr_dbic = (delta_bic.quantile(0.25), delta_bic.quantile(0.75))
    
    # Raftery (1995) evidence tiers
    weak_oqs = ((delta_bic >= 0) & (delta_bic < 2)).sum()
    pos_oqs = ((delta_bic >= 2) & (delta_bic < 6)).sum()
    strong_oqs = ((delta_bic >= 6) & (delta_bic < 10)).sum()
    vstrong_oqs = (delta_bic >= 10).sum()
    
    weak_ctmc = ((delta_bic < 0) & (delta_bic > -2)).sum()
    pos_ctmc = ((delta_bic <= -2) & (delta_bic > -6)).sum()
    strong_ctmc = ((delta_bic <= -6) & (delta_bic > -10)).sum()
    vstrong_ctmc = (delta_bic <= -10).sum()
    
    bound_hits = df_res['BoundHit'].sum()
    
    print("\n" + "=" * 70)
    print("KRAJBICH ET AL. (2010) BENCHMARK VALIDATION RESULTS")
    print(f"Total Trials Analyzed: {n_trials} across {len(subjects)} subjects")
    print(f"OQS4 Wins by BIC: {oqs_wins}/{n_trials} ({oqs_wins/n_trials*100:.1f}%)")
    print(f"CTMC Wins by BIC: {ctmc_wins}/{n_trials} ({ctmc_wins/n_trials*100:.1f}%)")
    print(f"Median Delta BIC (CTMC - OQS4): {median_dbic:.2f} [IQR: {iqr_dbic[0]:.2f}, {iqr_dbic[1]:.2f}]")
    print(f"Mean SSE: OQS4 = {df_res['SSE_OQS4'].mean():.4f} vs CTMC = {df_res['SSE_CTMC'].mean():.4f}")
    print(f"Evidence for OQS4: Weak [0,2) = {weak_oqs}, Positive [2,6) = {pos_oqs}, Strong [6,10) = {strong_oqs}, Very Strong >=10 = {vstrong_oqs}")
    print(f"Evidence for CTMC: Weak (-2,0) = {weak_ctmc}, Positive (-6,-2] = {pos_ctmc}, Strong (-10,-6] = {strong_ctmc}, Very Strong <=-10 = {vstrong_ctmc}")
    print(f"Boundary Parameter Hits: {bound_hits}/{n_trials} ({bound_hits/n_trials*100:.1f}%)")
    print("=" * 70)

if __name__ == '__main__':
    run_krajbich_validation()
