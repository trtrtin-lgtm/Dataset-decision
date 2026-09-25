# -*- coding: utf-8 -*-
"""
Bayesian Informative Priors vs Flat Priors Analysis on Empirical Eye-tracking Data (N=82)
Uses empirical calibrated priors derived from parameter recovery simulation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from modules.data_processor import solve_oqs_model

def run_bayesian_priors_analysis():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = os.path.join(ROOT, 'results', 'OQS_Run_20260923_213629_FIXED', 'Data_Excel', 'OQS_Results_Auto.xlsx')
    
    print(f"Loading empirical data from: {excel_path}")
    xls = pd.ExcelFile(excel_path)
    df_prob = pd.read_excel(xls, 'Prob_5bins')
    df_fit = pd.read_excel(xls, 'OQS_Fit_Results')
    
    # Calibrated empirical priors derived from simulation / recovery:
    # d in [0.5, 2.5] -> mu=1.5, sigma=0.6
    # gamma in [0.2, 2.5] -> mu=1.35, sigma=0.65
    # noise level sigma_noise = 0.04 -> variance = 0.0016
    mu_d, std_d = 1.50, 0.60
    mu_g1, std_g1 = 1.35, 0.65
    mu_g2, std_g2 = 1.35, 0.65
    sigma_noise = 0.04
    lambda_reg = 2.0 * (sigma_noise ** 2) # Bayesian negative log-posterior scaling
    
    bounds = [(0.01, 5.0), (1e-6, 5.0), (1e-6, 5.0)]
    
    results = []
    
    trials_grouped = df_prob.groupby(['subject', 'TrialID', 'no'])
    print(f"Total trials to evaluate: {len(trials_grouped)}")
    
    for (subj, tr_id, no_val), group in trials_grouped:
        t_eval = group['t_bin_center_sec'].values
        p11_true = group['Prob_Op1'].values
        p33_true = group['Prob_Op2'].values
        
        if len(t_eval) < 3:
            continue
            
        # Objective 1: Flat Prior (Pure MLE / SSE)
        def obj_flat(params):
            p11_pred, p33_pred = solve_oqs_model(params, t_eval, 'OQS4')
            return np.sum((p11_true - p11_pred)**2) + np.sum((p33_true - p33_pred)**2)
            
        # Objective 2: Informative Prior (MAP)
        def obj_map(params):
            sse = obj_flat(params)
            d, g1, g2 = params
            prior_penalty = (
                ((d - mu_d) / std_d)**2 +
                ((g1 - mu_g1) / std_g1)**2 +
                ((g2 - mu_g2) / std_g2)**2
            )
            return sse + lambda_reg * prior_penalty

        # Fit Flat Prior
        best_flat = None
        for _ in range(5):
            x0 = [np.random.uniform(*bounds[0]), np.random.uniform(*bounds[1]), np.random.uniform(*bounds[2])]
            res = minimize(obj_flat, x0, bounds=bounds, method='L-BFGS-B')
            if best_flat is None or res.fun < best_flat.fun:
                best_flat = res

        # Fit MAP Informative Prior
        best_map = None
        for _ in range(5):
            x0 = [np.random.uniform(*bounds[0]), np.random.uniform(*bounds[1]), np.random.uniform(*bounds[2])]
            res = minimize(obj_map, x0, bounds=bounds, method='L-BFGS-B')
            if best_map is None or res.fun < best_map.fun:
                best_map = res
                
        # Boundary hit definition: gamma < 0.02 or gamma > 4.98 or d < 0.02
        flat_d, flat_g1, flat_g2 = best_flat.x
        map_d, map_g1, map_g2 = best_map.x
        
        flat_bound = (flat_g1 < 0.02) or (flat_g1 > 4.98) or (flat_g2 < 0.02) or (flat_g2 > 4.98) or (flat_d < 0.02)
        map_bound = (map_g1 < 0.02) or (map_g1 > 4.98) or (map_g2 < 0.02) or (map_g2 > 4.98) or (map_d < 0.02)
        
        n_obs = len(t_eval) * 2
        sse_flat = float(best_flat.fun)
        # For MAP, calculate pure SSE on the fitted parameters
        sse_map_pure = float(obj_flat(best_map.x))
        
        bic_flat = n_obs * np.log(max(sse_flat / n_obs, 1e-12)) + 3 * np.log(n_obs)
        bic_map = n_obs * np.log(max(sse_map_pure / n_obs, 1e-12)) + 3 * np.log(n_obs)
        
        results.append({
            'subject': subj,
            'TrialID': tr_id,
            'no': no_val,
            'N_Obs': n_obs,
            'Flat_d': flat_d,
            'Flat_g1': flat_g1,
            'Flat_g2': flat_g2,
            'Flat_SSE': sse_flat,
            'Flat_BIC': bic_flat,
            'Flat_BoundHit': int(flat_bound),
            'MAP_d': map_d,
            'MAP_g1': map_g1,
            'MAP_g2': map_g2,
            'MAP_SSE': sse_map_pure,
            'MAP_BIC': bic_map,
            'MAP_BoundHit': int(map_bound),
        })
        
    df_res = pd.DataFrame(results)
    out_path = os.path.join(ROOT, 'results', 'Bayesian_Informative_Priors_Results.xlsx')
    df_res.to_excel(out_path, index=False)
    print(f"Saved results to: {out_path}")
    
    # Statistical Summary
    n_total = len(df_res)
    flat_hits = df_res['Flat_BoundHit'].sum()
    map_hits = df_res['MAP_BoundHit'].sum()
    
    print("=" * 60)
    print("BAYESIAN INFORMATIVE PRIORS VS FLAT PRIORS SUMMARY")
    print(f"Total Trials Evaluated: {n_total}")
    print(f"Flat Prior (MLE) Boundary Hits: {flat_hits}/{n_total} ({flat_hits/n_total*100:.1f}%)")
    print(f"Informative Prior (MAP) Boundary Hits: {map_hits}/{n_total} ({map_hits/n_total*100:.1f}%)")
    print(f"Boundary Hit Reduction: {(flat_hits - map_hits)/flat_hits * 100:.1f}% relative reduction")
    print(f"Mean SSE Flat: {df_res['Flat_SSE'].mean():.4f} vs MAP: {df_res['MAP_SSE'].mean():.4f}")
    print(f"Median Delta BIC (MAP - Flat): {(df_res['MAP_BIC'] - df_res['Flat_BIC']).median():.4f}")
    print("=" * 60)

if __name__ == '__main__':
    run_bayesian_priors_analysis()
