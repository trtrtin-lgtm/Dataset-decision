import pandas as pd
import sys
import matplotlib.pyplot as plt
import os
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

print("Khởi chạy module QQ Equality Analysis...")

fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Figures')
os.makedirs(fig_dir, exist_ok=True)

f1 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Khảo sát hành vi ra quyết định của sinh viên từ 18 đến 22 tuổi (1) (Câu trả lời).xlsx')
f2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Khảo sát hành vi ra quyết định của sinh viên từ 18 đến 22 tuổi (2) (Câu trả lời).xlsx')

try:
    df1 = pd.read_excel(f1)
    df2 = pd.read_excel(f2)
except Exception as e:
    print(f"Lỗi đọc file: {e}")
    sys.exit(1)

q_huy = 'Bạn thấy Huy là người như thế nào?'
q_kiemtra = 'Kiểm tra toàn bộ có ảnh hưởng đến bạn không?'

df1['Huy_Ans'] = df1[q_huy].apply(lambda x: 'Tốt' if 'thương' in str(x).lower() else 'Xấu')
df1['KT_Ans'] = df1[q_kiemtra].apply(lambda x: 'Có' if 'có,' in str(x).lower() else 'Không')
df2['Huy_Ans'] = df2[q_huy].apply(lambda x: 'Tốt' if 'thương' in str(x).lower() else 'Xấu')
df2['KT_Ans'] = df2[q_kiemtra].apply(lambda x: 'Có' if 'có,' in str(x).lower() else 'Không')

p2_yy_1 = len(df1[(df1['Huy_Ans']=='Tốt') & (df1['KT_Ans']=='Có')]) / len(df1)
p2_nn_1 = len(df1[(df1['Huy_Ans']=='Xấu') & (df1['KT_Ans']=='Không')]) / len(df1)

p2_yy_2 = len(df2[(df2['Huy_Ans']=='Tốt') & (df2['KT_Ans']=='Có')]) / len(df2)
p2_nn_2 = len(df2[(df2['Huy_Ans']=='Xấu') & (df2['KT_Ans']=='Không')]) / len(df2)

labels = ['Nhóm 1 (Hỏi A rồi B)', 'Nhóm 2 (Hỏi B rồi A)']
p_yy = [p2_yy_1, p2_yy_2]
p_nn = [p2_nn_1, p2_nn_2]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, p_yy, width, label='p(Đồng ý, Đồng ý)', color='#2ecc71')
rects2 = ax.bar(x + width/2, p_nn, width, label='p(Từ chối, Từ chối)', color='#e74c3c')

total_1 = p2_yy_1 + p2_nn_1
total_2 = p2_yy_2 + p2_nn_2
ax.plot([-0.3, 0.3], [total_1, total_1], color='black', linestyle='--', linewidth=2)
ax.plot([0.7, 1.3], [total_2, total_2], color='black', linestyle='--', linewidth=2, label='Tổng QQ (QQ Sum)')

ax.set_ylabel('Xác suất (Probability)')
ax.set_title('Hình 6: Khảo nghiệm Phương trình Đẳng thức Lượng tử (QQ Equality)\nChứng minh Tính Không Giao Hoán trong Khảo sát Tình huống 2', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0, 0.8)

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.4f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  
                textcoords='offset points',
                ha='center', va='bottom')

plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1)
fig_path = os.path.join(fig_dir, 'Fig6_QQEquality.png')
plt.savefig(fig_path, dpi=300)

# --- KIỂM ĐỊNH THỐNG KÊ (CHUẨN Q1) ---
from scipy.stats import fisher_exact

print(f"Tổng QQ Nhóm 1: {total_1:.4f}")
print(f"Tổng QQ Nhóm 2: {total_2:.4f}")
print(f"Sai lệch: {abs(total_1 - total_2):.4f}")

# Fisher's Exact Test: Kiểm tra sự khác biệt trong tỷ lệ Đồng ý/Từ chối giữa 2 nhóm
n1 = len(df1)
n2 = len(df2)
a = len(df1[(df1['Huy_Ans']=='Tốt') & (df1['KT_Ans']=='Có')])
b = n1 - a
c = len(df2[(df2['Huy_Ans']=='Tốt') & (df2['KT_Ans']=='Có')])
d_val = n2 - c
contingency_table = [[a, b], [c, d_val]]
odds_ratio, fisher_p = fisher_exact(contingency_table)

print(f"\n--- KIỂM ĐỊNH THỐNG KÊ ---")
print(f"Fisher's Exact Test:")
print(f"  Odds Ratio: {odds_ratio:.4f}")
print(f"  P-value: {fisher_p:.4f}")
if fisher_p < 0.05:
    print(f"  => Sự khác biệt giữa 2 nhóm có ý nghĩa thống kê (p < 0.05)!")
    
else:
    print(f"  => Sai lệch chưa đạt ý nghĩa thống kê (p >= 0.05).")
    
    print(f"  => Cần tăng cỡ mẫu (N > 100 mỗi nhóm) để đạt statistical power đủ mạnh.")

print("Plot saved to", fig_path)
