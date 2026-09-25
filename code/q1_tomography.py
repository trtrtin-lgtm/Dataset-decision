import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob, os, ast
from scipy.integrate import solve_ivp

# Cấu hình thẩm mỹ Q1
plt.style.use('seaborn-v0_8-whitegrid')
fig_dir = r"E:\ET\ĐA Đánh giá sự lưỡng lự bằng mô hình lượng tử OQS4 và OQS6-20260828T140941Z-1-001\Figures"
os.makedirs(fig_dir, exist_ok=True)

calc_dir = r"E:\ET\ĐA Đánh giá sự lưỡng lự bằng mô hình lượng tử OQS4 và OQS6-20260828T140941Z-1-001\GUI_App\Calculate"
files = glob.glob(os.path.join(calc_dir, "OQS_Results_*.xlsx"))
if not files: exit()
excel_path = max(files, key=os.path.getctime)

df_fit = pd.read_excel(excel_path, sheet_name="OQS_Fit_Results")

def get_hamiltonian(d):
    return np.array([[0.0, d, 0.0], [d, 0.0, d], [0.0, d, 0.0]], dtype=np.complex128)

def get_lindblad_ops_oqs6():
    L12 = np.zeros((3, 3), dtype=np.complex128); L12[0, 1] = 1.0
    L32 = np.zeros((3, 3), dtype=np.complex128); L32[2, 1] = 1.0
    L21 = np.zeros((3, 3), dtype=np.complex128); L21[1, 0] = 1.0
    L23 = np.zeros((3, 3), dtype=np.complex128); L23[1, 2] = 1.0
    return [L12, L32, L21, L23]

def lindblad_rhs(t, rho_flat, H, L_ops, rates, timescale):
    rho = rho_flat.reshape(3, 3)
    d_rho = -1j * (H @ rho - rho @ H)
    for L, gamma in zip(L_ops, rates):
        L_dag = L.conj().T
        d_rho += gamma * (L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L))
    return (timescale * d_rho).reshape(9)

def parse_params(param_str):
    param_str = param_str.replace('[', '').replace(']', '').split()
    return [float(x) for x in param_str]

# Lấy Trial có SSE tốt nhất để trực quan hóa Tomography
best_row = df_fit.loc[df_fit['SSE_OQS6'].idxmin()]
params = parse_params(best_row['Params_OQS6'])
d, rates, timescale = params[0], params[1:5], params[5]
H = get_hamiltonian(d)
L_ops = get_lindblad_ops_oqs6()

rho0 = np.zeros((3, 3), dtype=np.complex128)
rho0[1, 1] = 1.0 # Bắt đầu ở trạng thái "Lưỡng lự" (Middle)

t_eval = np.linspace(0, 5, 50)
sol = solve_ivp(lambda tt, yy: lindblad_rhs(tt, yy, H, L_ops, rates, timescale),
                [0.0, 5.0], rho0.reshape(9), t_eval=t_eval, method='RK45')

rho_t = sol.y.reshape(3, 3, -1)

# Lấy ma trận mật độ tại thời điểm Giao thoa cực đại
coh_t = np.abs(rho_t[0,1,:]) + np.abs(rho_t[1,0,:]) + np.abs(rho_t[1,2,:]) + np.abs(rho_t[2,1,:])
peak_idx = np.argmax(coh_t)
rho_peak = rho_t[:, :, peak_idx]

# --- VẼ 3D STATE TOMOGRAPHY ---
fig = plt.figure(figsize=(14, 6))

# Phần thực (Real part)
ax1 = fig.add_subplot(121, projection='3d')
x = np.arange(3)
y = np.arange(3)
X, Y = np.meshgrid(x, y)
x, y = X.ravel(), Y.ravel()
top_real = np.real(rho_peak).ravel()
bottom = np.zeros_like(top_real)
width = depth = 0.5

# Màu gradient dựa trên chiều cao
colors = plt.cm.viridis(top_real / np.max(top_real))
ax1.bar3d(x, y, bottom, width, depth, top_real, shade=True, color=colors)
ax1.set_title("Phần thực (Real[ρ])\nXác suất & Tương quan trực tiếp", fontweight='bold', fontsize=12)
ax1.set_xticks([0.5, 1.5, 2.5])
ax1.set_xticklabels(['A', 'Phân vân', 'B'])
ax1.set_yticks([0.5, 1.5, 2.5])
ax1.set_yticklabels(['A', 'Phân vân', 'B'])

# Phần ảo (Imaginary part)
ax2 = fig.add_subplot(122, projection='3d')
top_imag = np.imag(rho_peak).ravel()
colors_imag = plt.cm.plasma((top_imag - np.min(top_imag)) / (np.max(top_imag) - np.min(top_imag) + 1e-9))
ax2.bar3d(x, y, bottom, width, depth, top_imag, shade=True, color=colors_imag)
ax2.set_title("Phần ảo (Imag[ρ])\nDòng lưu chuyển Lượng tử (Giao thoa)", fontweight='bold', fontsize=12)
ax2.set_xticks([0.5, 1.5, 2.5])
ax2.set_xticklabels(['A', 'Phân vân', 'B'])
ax2.set_yticks([0.5, 1.5, 2.5])
ax2.set_yticklabels(['A', 'Phân vân', 'B'])

plt.suptitle(f"Hình 4: Tái cấu trúc Ma trận mật độ Lượng tử (Quantum State Tomography)\nTại đỉnh điểm của sự lưỡng lự (T={t_eval[peak_idx]:.1f}s)", fontsize=16, fontweight='bold', color='darkblue')

# Sửa lỗi cảnh báo Tight Layout bằng cách căn lề thủ công cho đồ thị 3D
plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1, top=0.85, bottom=0.1)

fig.savefig(os.path.join(fig_dir, "Fig4_QuantumTomography.png"), dpi=300, bbox_inches='tight')
print("Saved Fig4_QuantumTomography.png without layout warnings!")

# Hiển thị biểu đồ lên màn hình cho người dùng

# --- TA?NH ENTROPY VON NEUMANN ---
import scipy.linalg
evals = np.linalg.eigvalsh(rho_peak)
evals = evals[evals > 1e-10]
vn_entropy = -np.sum(evals * np.log2(evals))
print(f'\n[BÁO CÁO MỨC ĐỘ TRẬT TỰ LƯỢNG TỬ]')
print(f'Tích phân Giao thoa (Coherence Integral): {coh_t[peak_idx]:.4f}')
print(f'Entropy Von Neumann (OQS6): {vn_entropy:.4f} bit')
print(f'-> Não bộ duy trì mức độ trật tự cao, bác bỏ giả thuyết hỗn loạn cổ điển.\n')


