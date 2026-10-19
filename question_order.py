import numpy as np
from scipy.stats import fisher_exact

def calc_qq_joint_probabilities(choice_a, choice_b):
    a = np.asarray(choice_a)
    b = np.asarray(choice_b)
    n = len(a)
    p_11 = float(np.sum((a == 1) & (b == 1))) / n
    p_00 = float(np.sum((a == 0) & (b == 0))) / n
    return p_11, p_00

def calc_q_statistic(prob_ab, prob_ba):
    p_11_ab, p_00_ab = prob_ab
    p_11_ba, p_00_ba = prob_ba
    return float((p_11_ab + p_00_ab) - (p_11_ba + p_00_ba))

def run_fisher_order_test(n_yes_order1, n_no_order1, n_yes_order2, n_no_order2):
    table = [[n_yes_order1, n_no_order1],
             [n_yes_order2, n_no_order2]]
    oddsratio, pvalue = fisher_exact(table)
    return float(oddsratio), float(pvalue)
