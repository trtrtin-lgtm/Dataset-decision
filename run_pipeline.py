import os
import numpy as np
import pandas as pd
from data_processing import clean_pupil_signal, pfe_correction, response_locked_binning, check_quality_gate
from model_fitting import fit_oqs3, fit_ctmc4, calc_pseudo_bic
from multimodal_metrics import count_gaze_switches, calc_late_pupil_dilation, calc_dci, classify_hesitation_level
from question_order import calc_qq_joint_probabilities, calc_q_statistic, run_fisher_order_test

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    trials_path = os.path.join(data_dir, "trials.csv")
    if os.path.exists(trials_path):
        trials_df = pd.read_csv(trials_path)
        print(f"Loaded trials: {len(trials_df)} rows")
    t_eval = np.linspace(0.1, 2.9, 15)
    synthetic_p1 = 0.2 + 0.15 * (t_eval / 3.0)
    synthetic_p2 = 0.35 + 0.10 * (t_eval / 3.0)
    fit_oqs = fit_oqs3(synthetic_p1, synthetic_p2, t_eval, n_starts=5)
    fit_ctmc = fit_ctmc4(synthetic_p1, synthetic_p2, t_eval, n_starts=5)
    print(f"OQS3 fit SSE: {fit_oqs['sse']:.4f}, BIC: {fit_oqs['bic']:.2f}")
    print(f"CTMC4 fit SSE: {fit_ctmc['sse']:.4f}, BIC: {fit_ctmc['bic']:.2f}")
    dci = calc_dci(z_switches=1.2, z_mean_pupil=0.8, z_delta_pupil=0.5)
    level = classify_hesitation_level(dci)
    print(f"Computed DCI: {dci:.2f} (Hesitation Level: {level})")

if __name__ == "__main__":
    main()
