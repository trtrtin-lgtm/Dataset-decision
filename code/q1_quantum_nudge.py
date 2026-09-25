import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

print("Khởi chạy Kỹ thuật Điều khiển Nhận thức (Quantum Cognitive Control)...")

fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Figures')
os.makedirs(fig_dir, exist_ok=True)

def get_hamiltonian(d):
    return np.array([[0.0, d, 0.0], [d, 0.0, d], [0.0, d, 0.0]], dtype=np.complex128)

def get_lindblad_ops():
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

# Sinh viên Tê Liệt Zeno: d nhỏ (0.2), gamma nhỏ (0.01) -> Bị kẹt ở sự phân vân
d = 0.2
rates = [0.01, 0.01, 0.01, 0.01]
timescale = 1.0
H = get_hamiltonian(d)
L_ops = get_lindblad_ops()

rho0 = np.zeros((3, 3), dtype=np.complex128)
rho0[1, 1] = 1.0 

# Pha 1: Tự nhiên 0 -> 2s
t1 = np.linspace(0, 2.0, 100)
sol1 = solve_ivp(lambda tt, yy: lindblad_rhs(tt, yy, H, L_ops, rates, timescale),
                 [0.0, 2.0], rho0.reshape(9), t_eval=t1, method='RK45')
rho_t1 = sol1.y.reshape(3, 3, -1)
prob_decided_1 = np.real(rho_t1[0, 0, :]) + np.real(rho_t1[2, 2, :])

# Cú huých Unitary (Quantum Nudge): Tráo đổi trạng thái "Phân vân" (1) sang "Đồng ý" (0)
U = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
], dtype=np.complex128)

rho_mid = rho_t1[:, :, -1]
rho_nuged = U @ rho_mid @ U.conj().T

# Pha 2A: Có Can Thiệp
t2 = np.linspace(2.0, 5.0, 150)
sol2_nuged = solve_ivp(lambda tt, yy: lindblad_rhs(tt, yy, H, L_ops, rates, timescale),
                       [2.0, 5.0], rho_nuged.reshape(9), t_eval=t2, method='RK45')
rho_t2_nuged = sol2_nuged.y.reshape(3, 3, -1)
prob_decided_2_nuged = np.real(rho_t2_nuged[0, 0, :]) + np.real(rho_t2_nuged[2, 2, :])

# Pha 2B: Không Can Thiệp
sol2_natural = solve_ivp(lambda tt, yy: lindblad_rhs(tt, yy, H, L_ops, rates, timescale),
                         [2.0, 5.0], rho_mid.reshape(9), t_eval=t2, method='RK45')
rho_t2_natural = sol2_natural.y.reshape(3, 3, -1)
prob_decided_2_natural = np.real(rho_t2_natural[0, 0, :]) + np.real(rho_t2_natural[2, 2, :])

t_full = np.concatenate((t1, t2))
prob_full_natural = np.concatenate((prob_decided_1, prob_decided_2_natural))
prob_full_nuged = np.concatenate((prob_decided_1, prob_decided_2_nuged))

plt.figure(figsize=(10, 6))
plt.plot(t_full, prob_full_natural*100, label="Không can thiệp (Mắc kẹt Zeno Paralysis)", color='gray', linestyle='--', linewidth=2.5)
plt.plot(t_full, prob_full_nuged*100, label="Có 'Cú huých Lượng tử' (Unitary UX Nudge)", color='red', linewidth=3)

plt.axvline(x=2.0, color='black', linestyle=':', linewidth=2)
plt.scatter([2.0], [prob_decided_1[-1]*100], color='red', s=100, zorder=5)
plt.annotate("Kích hoạt UI/UX Nudge\n(Toán tử Unitary $U$)",
             xy=(2.0, prob_decided_1[-1]*100), xytext=(0.5, prob_decided_1[-1]*100 + 30),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, fontweight='bold', bbox=dict(boxstyle="round", fc="yellow", alpha=0.3))

plt.title("Hình 8: Kỹ thuật Điều khiển Nhận thức - Cú huých Lượng tử (Quantum Nudge)\nPhá vỡ 'Tê liệt Zeno' bằng Toán học Lượng tử", fontweight='bold', fontsize=14)
plt.xlabel("Thời gian suy nghĩ (giây)", fontsize=12)
plt.ylabel("Xác suất Chốt Quyết Định (%)", fontsize=12)
plt.grid(True, alpha=0.4)
plt.legend(loc="lower right", fontsize=11)
plt.ylim(0, 105)
plt.xlim(0, 5)

fig_path = os.path.join(fig_dir, 'Fig8_QuantumNudge.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')

print(f"Mô phỏng thành công! Tại t=2.0s, não bộ mắc kẹt với xác suất chốt quyết định chỉ {prob_decided_1[-1]*100:.1f}%.")
print(f"Ngay khi kích hoạt Pop-up (Cú huých Unitary), xác suất sụp đổ vọt thẳng lên {prob_decided_2_nuged[0]*100:.1f}%.")
print("Đột phá: Chúng ta không chỉ ĐỌC tâm trí, chúng ta có thể ĐIỀU KHIỂN và BẺ LÁI tâm trí bằng Toán học Lượng tử!")
print(f"Đồ thị lưu tại: {fig_path}")

# ==============================================================================
# --- VẼ 3D TOMOGRAPHY SO SÁNH (TRƯỚC VÀ SAU KHI NUDGE) ---
# ==============================================================================
fig3d = plt.figure(figsize=(14, 6))

# Subplot 1: Trước Cú Huých (Zeno Paralysis)
ax1 = fig3d.add_subplot(121, projection='3d')
xpos, ypos = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos)
dx = dy = 0.5
dz = np.real(rho_mid).ravel()

colors = ['salmon' if (x==1 and y==1) else 'lightblue' for x, y in zip(xpos, ypos)]
ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True, alpha=0.9)
ax1.set_title("Trạng thái Não TRƯỚC Cú Huých\n(Mắc kẹt Lưỡng lự - Đỉnh Đỏ giữa)", fontweight='bold')
ax1.set_zlim(0, 1)
ax1.set_xticks([0.5, 1.5, 2.5])
ax1.set_xticklabels(['ĐồngÝ', 'PhânVân', 'TừChối'])
ax1.set_yticks([0.5, 1.5, 2.5])
ax1.set_yticklabels(['ĐồngÝ', 'PhânVân', 'TừChối'])

# Subplot 2: Sau Cú Huých (Quantum Nudge Applied)
ax2 = fig3d.add_subplot(122, projection='3d')
dz_nuged = np.real(rho_nuged).ravel()
colors_nuged = ['lightgreen' if (x==0 and y==0) else 'lightblue' for x, y in zip(xpos, ypos)]
ax2.bar3d(xpos, ypos, zpos, dx, dy, dz_nuged, color=colors_nuged, shade=True, alpha=0.9)
ax2.set_title("Trạng thái Não SAU Cú Huých Unitary\n(Ép sụp đổ - Đỉnh Xanh góc)", fontweight='bold')
ax2.set_zlim(0, 1)
ax2.set_xticks([0.5, 1.5, 2.5])
ax2.set_xticklabels(['ĐồngÝ', 'PhânVân', 'TừChối'])
ax2.set_yticks([0.5, 1.5, 2.5])
ax2.set_yticklabels(['ĐồngÝ', 'PhânVân', 'TừChối'])

fig3d.suptitle("Hình 9: Trực quan 3D Ma trận Mật độ bị bẻ gãy bởi Cú Huých Lượng Tử", fontsize=16, fontweight='bold', color='darkred')
plt.tight_layout()
fig9_path = os.path.join(fig_dir, 'Fig9_NudgeTomography.png')
plt.savefig(fig9_path, dpi=300)
print(f"Đã xuất 3D Tomography So sánh tại: {fig9_path}")
