import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import sys
import glob

sys.stdout.reconfigure(encoding='utf-8')

print("Khởi chạy module Phân cụm K-Means Hồ sơ Nhận thức trên dữ liệu OQS THẬT...")

fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Figures')
os.makedirs(fig_dir, exist_ok=True)
calc_dir = os.path.join(os.path.dirname(__file__), 'Calculate')

files = glob.glob(os.path.join(calc_dir, "OQS_Results_*.xlsx"))
if not files:
    print("Chưa có file OQS_Results. Hãy chạy Phân tích trước.")
    sys.exit(1)

excel_path = max(files, key=os.path.getctime)
df_fit = pd.read_excel(excel_path, sheet_name="OQS_Fit_Results")

d_list = []
gamma_list = []

for idx, row in df_fit.iterrows():
    try:
        param_str = row['Params_OQS6'].replace('[', '').replace(']', '').split()
        d_val = float(param_str[0])
        gamma_val = float(param_str[5])
        d_list.append(d_val)
        gamma_list.append(gamma_val)
    except:
        pass

if len(d_list) < 2:
    print("Dữ liệu không đủ để phân cụm. Hãy chắc chắn OQS6 đã hội tụ.")
    sys.exit(1)

data = pd.DataFrame({'Oscillation_d': d_list, 'Decoherence_gamma': gamma_list})

# Xử lý outliers để K-Means không bị nhiễu do các fits chưa chuẩn
# Chỉ lấy các giá trị nằm trong ngưỡng hợp lý (vd d < 5, gamma < 5)
data = data[(data['Oscillation_d'] < 10) & (data['Decoherence_gamma'] < 10)]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=2, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8,6))

# Nhóm màu tùy thuộc vào tâm cụm để giữ logic: Nhóm có d cao, gamma thấp (Đỏ) - Nhóm có d thấp, gamma cao (Xanh)
centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)

if centroids[0, 0] > centroids[1, 0]:
    cluster_mapping = {0: 1, 1: 0}
else:
    cluster_mapping = {0: 0, 1: 1}

data['Cluster_Mapped'] = data['Cluster'].map(cluster_mapping)

colors = ['#3498db' if c == 0 else '#e74c3c' for c in data['Cluster_Mapped']]
plt.scatter(data['Oscillation_d'], data['Decoherence_gamma'], c=colors, s=100, alpha=0.7, edgecolors='k')

plt.scatter(centroids[:, 0], centroids[:, 1], c='yellow', s=200, marker='*', edgecolors='k', label='Centroids (Trọng tâm)')

plt.xlabel('Tốc độ Dao động Lượng tử (d) - Suy nghĩ đa chiều', fontsize=12)
plt.ylabel('Tốc độ Suy biến (gamma) - Chốt quyết định', fontsize=12)
plt.title('Hình 5: Phân cụm Hồ sơ Nhận thức bằng Thuật toán K-Means\n(Dữ liệu thực nghiệm Eye-tracking + OQS6)', fontweight='bold', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Nhóm 1: Trực giác nhanh (d thấp, gamma cao)', markerfacecolor='#3498db', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Nhóm 2: Phân tích sâu (d cao, gamma thấp)', markerfacecolor='#e74c3c', markersize=10),
                   Line2D([0], [0], marker='*', color='w', label='Centroids', markerfacecolor='yellow', markeredgecolor='k', markersize=15)]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
fig_path = os.path.join(fig_dir, 'Fig5_CognitiveClusters.png')
plt.savefig(fig_path, dpi=300)
print("Plot saved to", fig_path)

