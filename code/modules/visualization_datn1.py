import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def show_datn1_plots(excel_path):
    try:
        df_trans = pd.read_excel(excel_path, sheet_name="Transitions_trial_5bins")
        df_prob = pd.read_excel(excel_path, sheet_name="Prob_5bins")
    except Exception as e:
        print("Không thể đọc file hoặc thiếu dữ liệu sheet (Prob_5bins, Transitions):", e)
        return

    # Nếu DataFrame không trống, thiết lập giao diện vẽ
    sns.set_style("whitegrid")
    
    # Tạo Figure với 2 đồ thị, sắp xếp y hệt mục 4.2.1 và 4.2.5 của DATN1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.canvas.manager.set_window_title("Đối chiếu Kết quả Luận văn DATN1")
    
    # --- Hình 1: Xác suất chuyển đổi (Transition Probability) theo thời gian ---
    if not df_trans.empty and 'Prob_Transition' in df_trans.columns:
        prob_over_time = df_trans.groupby("bin")["Prob_Transition"].mean()
        
        ax1.plot(prob_over_time.index, prob_over_time.values, marker='o', 
                 color='#1f77b4', linewidth=2.5, markersize=8)
        
        ax1.set_title("Hình 4.2.1: Xác suất chuyển đổi ánh nhìn theo thời gian", fontsize=13, fontweight='bold')
        ax1.set_xlabel("Time Bin (Chuẩn hoá theo thời gian)", fontsize=11)
        ax1.set_ylabel("Xác suất chuyển đổi (Prob_Transition)", fontsize=11)
        ax1.set_xticks([1, 2, 3, 4, 5])
        ax1.set_ylim(bottom=0)
    else:
        ax1.text(0.5, 0.5, "Không có dữ liệu Prob_Transition", ha='center')

    # --- Hình 2: Chồng ghép Ánh nhìn & Đồng tử (Overlay Dynamics) ---
    if not df_trans.empty and not df_prob.empty and 'PupilAbsDiff_Choice' in df_prob.columns:
        prob_t = df_trans.groupby("bin")["Prob_Transition"].mean()
        pupil_t = df_prob.groupby("bin")["PupilAbsDiff_Choice"].mean()
        
        # Trục 1: Ánh nhìn
        line1 = ax2.plot(prob_t.index, prob_t.values, marker='o', 
                         color='#ff7f0e', linewidth=2.5, markersize=8, label="Xác suất chuyển đổi (Ánh nhìn)")
        
        # Trục 2: Đồng tử
        ax2_twin = ax2.twinx()
        line2 = ax2_twin.plot(pupil_t.index, pupil_t.values, marker='s', 
                              color='#2ca02c', linewidth=2.5, markersize=8, linestyle='--', label="Mức thay đổi đồng tử (|ΔP|)")
        
        ax2.set_title("Hình 4.2.5: Chồng ghép mức chuyển đổi ánh nhìn và đồng tử", fontsize=13, fontweight='bold')
        ax2.set_xlabel("Time Bin (Chuẩn hoá theo thời gian)", fontsize=11)
        ax2.set_ylabel("Xác suất chuyển đổi (Prob_Transition)", fontsize=11)
        ax2_twin.set_ylabel("Độ biến thiên tuyệt đối đồng tử (|ΔP|)", fontsize=11)
        ax2.set_xticks([1, 2, 3, 4, 5])
        
        # Gộp Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left', frameon=True)
    else:
        ax2.text(0.5, 0.5, "Không có dữ liệu PupilAbsDiff_Choice", ha='center')

    plt.tight_layout()
    plt.show()
