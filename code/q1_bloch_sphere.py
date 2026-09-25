import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình đường dẫn lưu ảnh
fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Figures')
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, 'Fig11_BlochSphere.png')

fig = plt.figure(figsize=(14, 7))

# Tính toán mặt cầu cơ sở (Khung lưới)
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)

# ==========================================
# 1. TÂM TRÍ CỔ ĐIỂN (Logic Công tắc Đèn)
# ==========================================
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_wireframe(x, y, z, color="lightgray", alpha=0.15)
ax1.plot([0,0], [0,0], [-1.2, 1.2], color="black", linestyle="--", linewidth=1) # Trục Z
ax1.plot([-1.2, 1.2], [0,0], [0,0], color="black", linestyle="--", linewidth=1) # Trục X
ax1.plot([0,0], [-1.2, 1.2], [0,0], color="black", linestyle="--", linewidth=1) # Trục Y

# Cổ điển chỉ nảy bật giữa 2 cực (Đồng ý = +1, Từ chối = -1)
# Vẽ đường nhảy cóc ziczac thẳng đứng
ax1.scatter([0], [0], [1], color="green", s=200, label="Trạng thái: ĐỒNG Ý", edgecolors='black')
ax1.scatter([0], [0], [-1], color="red", s=200, label="Trạng thái: TỪ CHỐI", edgecolors='black')
ax1.plot([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 1, -1, 1, -1], color="red", linewidth=4, linestyle=":", alpha=0.7)

ax1.set_title("TÂM TRÍ CỔ ĐIỂN (CLASSICAL MIND)\nLogic 0 hoặc 1. Lưỡng lự là nhảy cóc rạc rạc.", fontweight="bold", fontsize=12)
ax1.axis('off')
ax1.text2D(0.5, 0.05, "Giống như: CÔNG TẮC BẬT/TẮT", transform=ax1.transAxes, ha='center', fontsize=11, color="red", fontweight="bold")


# ==========================================
# 2. TÂM TRÍ LƯỢNG TỬ (Quả cầu Bloch)
# ==========================================
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_wireframe(x, y, z, color="lightblue", alpha=0.2)
ax2.plot([0,0], [0,0], [-1.2, 1.2], color="black", linestyle="--", linewidth=1)
ax2.plot([-1.2, 1.2], [0,0], [0,0], color="black", linestyle="--", linewidth=1)
ax2.plot([0,0], [-1.2, 1.2], [0,0], color="black", linestyle="--", linewidth=1)

# Lượng tử là một QUỸ ĐẠO HÀM SÓNG xoắn ốc liên tục trong không gian 3D
t = np.linspace(0, 4*np.pi, 200)
z_q = np.linspace(-1, 1, 200) # Đi từ dưới (Từ chối) lên trên (Đồng ý)
r = np.sqrt(1 - z_q**2)
x_q = r * np.cos(t)
y_q = r * np.sin(t)

# Quỹ đạo suy nghĩ xoay mượt mà
ax2.plot(x_q, y_q, z_q, color="blue", linewidth=3, label="Quỹ đạo Chồng chập (Hàm sóng)")
# Mũi tên Vector trạng thái hiện tại (Vector Bloch)
current_idx = 140
ax2.quiver(0, 0, 0, x_q[current_idx], y_q[current_idx], z_q[current_idx], color='purple', linewidth=4, arrow_length_ratio=0.15)
ax2.scatter(x_q[-1], y_q[-1], z_q[-1], color="green", s=200, edgecolors='black', label="Sụp đổ về ĐỒNG Ý")

ax2.set_title("TÂM TRÍ LƯỢNG TỬ (QUANTUM MIND)\nTâm trí là một không gian 3D. Suy nghĩ là quỹ đạo xoay.", fontweight="bold", fontsize=12)
ax2.axis('off')
ax2.text2D(0.5, 0.05, "Giống như: CON QUAY HỒI CHUYỂN", transform=ax2.transAxes, ha='center', fontsize=11, color="blue", fontweight="bold")

plt.suptitle("Hình 11: BỨC TRANH TRỰC QUAN NHẤT VỀ SỰ ĐỘT PHÁ CỦA DỰ ÁN", fontsize=18, color="darkred", fontweight="bold")
plt.tight_layout()

plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Thành công vẽ Hình cầu Bloch tại: {fig_path}")
