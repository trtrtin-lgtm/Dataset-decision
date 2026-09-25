import pandas as pd
import numpy as np
import os
import ast
from scipy.stats import kruskal
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("--- NÂNG CẤP PHÂN TÍCH: ĐỐI TƯỢNG VÀ KÍCH THÍCH ---")

# 1. Đọc dữ liệu
try:
    df = pd.read_excel('Calculate/OQS_Results_Auto.xlsx', sheet_name='OQS_Fit_Results')
except Exception as e:
    print("Không tìm thấy dữ liệu:", e)
    sys.exit(1)

# Hàm giải mã w
def extract_w(param_str):
    if pd.isna(param_str): return np.nan
    param_str = str(param_str).strip('[]')
    parts = param_str.replace('\n', ' ').replace(',', ' ').split()
    if len(parts) >= 4:
        return float(parts[3])
    return np.nan

df['w'] = df['Params_Hybrid'].apply(extract_w)
df = df.dropna(subset=['w'])

if len(df) == 0:
    print("Không có dữ liệu w hợp lệ.")
    sys.exit(1)

# 2. Phân tích theo Kích thích (TrialID)
print("\n[A] PHÂN TÍCH THEO KÍCH THÍCH (SCENARIOS)")
trial_groups = [group['w'].values for name, group in df.groupby('TrialID')]
if len(trial_groups) > 1:
    stat, p_val = kruskal(*trial_groups)
    print(f"Kruskal-Wallis Test theo TrialID: H = {stat:.2f}, p-value = {p_val:.4f}")
    if p_val < 0.05:
        print("=> CÓ sự khác biệt có ý nghĩa thống kê về mức độ Lượng tử giữa các kích thích khác nhau!")
    else:
        print("=> Chưa đủ cơ sở thống kê để khẳng định kích thích tạo ra khác biệt lớn (p > 0.05).")

trial_summary = df.groupby('TrialID')['w'].agg(['mean', 'median', 'count']).sort_values('mean')
print("\nXếp hạng Kích thích (Từ Lượng tử/Lai -> Cổ điển):")
print(trial_summary)

# 3. Phân loại Đối tượng (Subjects)
print("\n[B] PHÂN LOẠI ĐỐI TƯỢNG (COGNITIVE STYLES)")
subj_groups = [group['w'].values for name, group in df.groupby('subject')]
if len(subj_groups) > 1:
    stat_s, p_val_s = kruskal(*subj_groups)
    print(f"Kruskal-Wallis Test theo Subject: H = {stat_s:.2f}, p-value = {p_val_s:.4f}")
    if p_val_s < 0.05:
        print("=> CÓ sự khác biệt có ý nghĩa thống kê giữa các cá nhân!")
    else:
        print("=> Các cá nhân có sự khác biệt nhưng chưa đủ ý nghĩa thống kê ở cỡ mẫu này.")

# Gom cụm cá nhân (Clustering)
subj_summary = df.groupby('subject')['w'].agg(['mean', 'std', 'count']).fillna(0)
# Lọc các subject có ít nhất 3 trials để phân loại cho chính xác
subj_valid = subj_summary[subj_summary['count'] >= 2].copy()

if len(subj_valid) >= 2:
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    # Phân loại dựa trên w trung bình
    labels = kmeans.fit_predict(subj_valid[['mean']])
    subj_valid['Cluster'] = labels
    
    # Đặt tên cụm
    cluster_means = subj_valid.groupby('Cluster')['mean'].mean()
    hybrid_cluster = cluster_means.idxmin()
    classical_cluster = cluster_means.idxmax()
    
    subj_valid['Cognitive_Style'] = subj_valid['Cluster'].map({
        hybrid_cluster: 'Hybrid / Intuitive Thinkers',
        classical_cluster: 'Classical / Analytic Thinkers'
    })
    
    print(f"\nPhân loại được {len(subj_valid)} đối tượng thành 2 nhóm nhận thức:")
    print(subj_valid[['mean', 'Cognitive_Style']].sort_values('mean'))
    
    # 4. Vẽ biểu đồ
    fig_dir = r'../Figures'
    os.makedirs(fig_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: w by TrialID
    sns.boxplot(x='TrialID', y='w', data=df, ax=axes[0], palette="Set2")
    axes[0].set_title('Hybrid Weight (w) Across Different Stimuli (Trials)')
    axes[0].set_xlabel('Stimulus (Trial ID)')
    axes[0].set_ylabel('Hybrid Weight w (0=Quantum, 1=Classical)')
    
    # Plot 2: Clusters of Subjects
    sns.scatterplot(x='count', y='mean', hue='Cognitive_Style', data=subj_valid, ax=axes[1], s=100, palette="husl")
    axes[1].set_title('Classification of Subjects by Cognitive Style')
    axes[1].set_xlabel('Number of Valid Trials per Subject')
    axes[1].set_ylabel('Mean Hybrid Weight (w)')
    axes[1].set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'Fig13_Subject_Stimulus_Analysis.png'), dpi=300)
    print("\n=> Đã lưu biểu đồ phân tích vào Fig13_Subject_Stimulus_Analysis.png")

print("Hoàn tất phân tích mở rộng.")
