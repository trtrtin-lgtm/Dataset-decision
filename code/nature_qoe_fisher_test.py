import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.stdout.reconfigure(encoding='utf-8')

def run_fisher_test():
    n1, n2 = 86, 87
    k1 = int(round(0.50 * n1))
    c1 = n1 - k1
    k2 = int(round(0.619 * n2))
    c2 = n2 - k2
    
    contingency_table = [[k1, c1], [k2, c2]]
    oddsratio, pvalue = stats.fisher_exact(contingency_table)
    
    print("--- QOE FISHER EXACT TEST (SITUATION 2) ---")
    print(f"Nhóm 1: {k1} / {c1}")
    print(f"Nhóm 2: {k2} / {c2}")
    print(f"Odds Ratio: {oddsratio:.4f}")
    print(f"P-Value: {pvalue:.4f}")
    
    fig, ax = plt.subplots(figsize=(6,5))
    df = pd.DataFrame({
        'Group': ['Nhóm 1 (A->B)', 'Nhóm 1 (A->B)', 'Nhóm 2 (B->A)', 'Nhóm 2 (B->A)'],
        'Choice': ['Khai báo', 'Che giấu', 'Khai báo', 'Che giấu'],
        'Count': [k1, c1, k2, c2]
    })
    sns.barplot(x='Group', y='Count', hue='Choice', data=df, palette='viridis', ax=ax)
    ax.set_title(f'Macroscopic Survey (Situation 2)\nFisher Exact Test p = {pvalue:.4f} (Not Significant)')
    plt.tight_layout()
    plt.savefig('qoe_fisher_result.png', dpi=300)
    print("Saved plot to qoe_fisher_result.png")

if __name__ == "__main__":
    run_fisher_test()
