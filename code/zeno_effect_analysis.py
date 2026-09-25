import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import glob, os
from scipy.stats import pearsonr

# Tìm file Excel mới nhất (đường dẫn TƯƠNG ĐỐI, không hardcode)
calc_dir = os.path.join(os.path.dirname(__file__), 'Calculate')
files = glob.glob(os.path.join(calc_dir, "OQS_Results_*.xlsx"))
if not files: 
    print("Chưa có file dữ liệu. Hãy chạy phân tích trước.")
    exit()
excel_path = max(files, key=os.path.getctime)

df_trans = pd.read_excel(excel_path, sheet_name="Transitions_trial_5bins")
df_fit = pd.read_excel(excel_path, sheet_name="OQS_Fit_Results")

# Tính tổng số lượt chuyển đổi (Transitions) cho mỗi Trial
trial_transitions = df_trans.groupby(['subject', 'TrialID'])[['b12', 'b21']].sum().reset_index()
trial_transitions['Total_Jumps'] = trial_transitions['b12'] + trial_transitions['b21']

# QUAN TRỌNG: Trích xuất ĐÚNG biến vật lý — Tốc độ Decoherence (gamma trung bình)
# chứ KHÔNG PHẢI timescale (hệ số co giãn thời gian)
decoherence_data = []
for idx, row in df_fit.iterrows():
    subj = row['subject']
    tid = row['TrialID']
    try:
        params_str = row['Params_OQS6'].replace('[', '').replace(']', '').split()
        # OQS6 params: [d, gamma1, gamma2, gamma3, gamma4, timescale]
        # Tốc độ Decoherence = Trung bình của 4 hệ số Lindblad (params[1:5])
        gammas = [float(params_str[i]) for i in range(1, 5)]
        avg_gamma = np.mean(gammas)
        decoherence_data.append({'subject': subj, 'TrialID': tid, 'Avg_Decoherence_Rate': avg_gamma})
    except:
        pass

df_deco = pd.DataFrame(decoherence_data)
df_zeno = pd.merge(trial_transitions, df_deco, on=['subject', 'TrialID'])

# --- 1. TÍNH PEARSON TRUYỀN THỐNG ---
r, p = pearsonr(df_zeno['Total_Jumps'], df_zeno['Avg_Decoherence_Rate'])

# --- 2. TÍNH BOOTSTRAP (10,000 VÒNG LẶP) ĐỂ BẢO VỆ MÔ HÌNH TRƯỚC REVIEWER Q1 ---
np.random.seed(42) # Cố định seed để kết quả ổn định
n_iterations = 10000
bootstrapped_r = []
n_size = len(df_zeno)
jumps = df_zeno['Total_Jumps'].values
rates = df_zeno['Avg_Decoherence_Rate'].values

for i in range(n_iterations):
    indices = np.random.randint(0, n_size, n_size)
    r_boot, _ = pearsonr(jumps[indices], rates[indices])
    bootstrapped_r.append(r_boot)

ci_lower = np.percentile(bootstrapped_r, 2.5)
ci_upper = np.percentile(bootstrapped_r, 97.5)

# --- XUẤT KẾT QUẢ HIỂN THỊ LÊN GUI ---
print("[PHÂN TÍCH TIÊU CHUẨN BÁO CÁO Q1]")
print("-" * 50)
print(f"1. Phân tích Pearson tuyến tính (Raw):")
print(f"   - Biến X: Tổng số lần đảo mắt giữa 2 đáp án (Total_Jumps)")
print(f"   - Biến Y: Tốc độ Decoherence trung bình (avg gamma_1..4)")
print(f"   - Hệ số tương quan (r): {r:.4f}")
print(f"   - P-value: {p:.4f}")
if p < 0.05:
    print(f"   => Đạt mức ý nghĩa thống kê (p < 0.05)!")
else:
    print(f"   => Chưa đạt mức ý nghĩa tuyệt đối 0.05 (có thể do cỡ mẫu nhỏ).")
print("")
print(f"2. Phân tích Lấy mẫu lại Bootstrap (10,000 vòng lặp):")
print(f"   - Nhằm khử nhiễu và đánh giá độ bền vững của dữ liệu theo chuẩn Q1.")
print(f"   - Khoảng tin cậy 95% (95% CI): [{ci_lower:.4f}, {ci_upper:.4f}]")
print("-" * 50)
print("=> KẾT LUẬN:")
if ci_upper < 0:
    print("Khoảng tin cậy 95% hoàn toàn nằm ở miền ÂM (không cắt qua 0).")
    print("Có bằng chứng thống kê cho thấy: Tần suất quan trắc (đảo mắt) càng cao")
    print("thì tốc độ Decoherence càng THẤP → Phù hợp với giả thuyết Hiệu ứng Zeno Lượng tử.")
elif ci_lower > 0:
    print("Khoảng tin cậy 95% hoàn toàn nằm ở miền DƯƠNG.")
    print("Tần suất quan trắc tỷ lệ thuận với tốc độ Decoherence → Anti-Zeno Effect.")
else:
    print(f"Khoảng tin cậy 95% cắt qua 0 → Chưa đủ bằng chứng để kết luận.")
    print("Cần thu thập thêm dữ liệu để khẳng định chắc chắn Hiệu ứng Zeno.")
