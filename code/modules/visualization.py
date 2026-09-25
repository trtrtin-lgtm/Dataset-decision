import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_comparison_plots(excel_path):
    try:
        df_fit = pd.read_excel(excel_path, sheet_name="OQS_Fit_Results")
        df_trans = pd.read_excel(excel_path, sheet_name="Transitions_trial_5bins")
        df_prob = pd.read_excel(excel_path, sheet_name="Prob_5bins")
    except Exception as e:
        print("Cannot read excel file or missing sheets:", e)
        return

    # Map colors
    color_map = {
        'OQS4': '#F4C7D0',           # Pink
        'OQS6': '#D7DAB3',           # Beige
        'Classical_CTMC': '#4A6644'  # Dark Green
    }
    
    # Create a figure with 2x3 subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.canvas.manager.set_window_title("Phân tích Khoa học - Động lực học Lưỡng lự")
    
    ax1, ax2, ax3 = axs[0, 0], axs[0, 1], axs[0, 2]
    ax4, ax5, ax6 = axs[1, 0], axs[1, 1], axs[1, 2]
    
    # --- Plot 1: Pie chart of best models based on BIC ---
    # Hỗ trợ cả cột cũ (BestModel) và cột mới (BestModel_BIC)
    col_best = 'BestModel_BIC' if 'BestModel_BIC' in df_fit.columns else 'BestModel'
    best_counts = df_fit[col_best].value_counts()
    pie_colors = [color_map.get(m, '#000000') for m in best_counts.index]
    ax1.pie(best_counts, labels=best_counts.index, autopct='%1.1f%%', startangle=90, colors=pie_colors)
    ax1.set_title("Tỉ lệ Mô hình tối ưu nhất (Theo BIC)", fontweight="bold")
    
    # --- Plot 2: Average SSE for models (Bar Chart) ---
    avg_sse = df_fit[['SSE_OQS4', 'SSE_OQS6', 'SSE_CTMC']].mean()
    bar_labels = ['OQS4', 'OQS6', 'Classical_CTMC']
    bar_values = [avg_sse['SSE_OQS4'], avg_sse['SSE_OQS6'], avg_sse['SSE_CTMC']]
    bar_colors = [color_map[m] for m in bar_labels]
    
    bars = ax2.bar(bar_labels, bar_values, color=bar_colors, edgecolor='black')
    ax2.set_title("Trung bình Sai số Dự đoán (Mean SSE)", fontweight="bold")
    ax2.set_ylabel("SSE")
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.4f}', va='bottom', ha='center')
        
    # --- Plot 3: SSE Distribution (Boxplot) ---
    sse_data = [df_fit['SSE_OQS4'].dropna(), df_fit['SSE_OQS6'].dropna(), df_fit['SSE_CTMC'].dropna()]
    bplot = ax3.boxplot(sse_data, patch_artist=True, labels=bar_labels)
    ax3.set_title("Phân phối Sai số (SSE Boxplot)", fontweight="bold")
    ax3.set_ylabel("SSE")
    for patch, color in zip(bplot['boxes'], bar_colors):
        patch.set_facecolor(color)
        
    # --- Plot 4: Transition Frequencies ---
    if not df_trans.empty:
        mean_trans = df_trans[['w1', 'w2', 'b12', 'b21']].mean()
        trans_labels = ['w1 (Đứng Op1)', 'w2 (Đứng Op2)', 'b12 (Chuyển 1->2)', 'b21 (Chuyển 2->1)']
        trans_colors = ['#87CEFA', '#4682B4', '#FFA500', '#FF4500']
        
        bars4 = ax4.bar(trans_labels, mean_trans.values, color=trans_colors, edgecolor='black')
        ax4.set_title("Tần suất Chuyển trạng thái Trung bình (Per Bin)", fontweight="bold")
        ax4.set_ylabel("Số lượng")
        ax4.tick_params(axis='x', rotation=15)
        
        for bar in bars4:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'{yval:.2f}', va='bottom', ha='center')
            
    # --- Plot 5: Time-series Transition Probability ---
    if not df_trans.empty and 'Prob_Transition' in df_trans.columns:
        prob_over_time = df_trans.groupby("bin")["Prob_Transition"].mean()
        ax5.plot(prob_over_time.index, prob_over_time.values, marker='o', color='#FF4500', linewidth=2, label="Prob_Transition")
        ax5.set_title("Xác suất chuyển đổi ánh nhìn theo thời gian", fontweight="bold")
        ax5.set_xlabel("Time Bin")
        ax5.set_ylabel("Xác suất (Prob_Transition)")
        ax5.set_xticks([1,2,3,4,5])
        ax5.grid(True, linestyle='--', alpha=0.6)
        
    # --- Plot 6: Overlay Dynamics (Transition vs PupilAbsDiff_Choice) ---
    if not df_trans.empty and not df_prob.empty and 'PupilAbsDiff_Choice' in df_prob.columns:
        prob_t = df_trans.groupby("bin")["Prob_Transition"].mean()
        pupil_t = df_prob.groupby("bin")["PupilAbsDiff_Choice"].mean()
        
        ax6.plot(prob_t.index, prob_t.values, marker='o', color='#FF4500', linewidth=2, label="Prob_Transition (Ánh nhìn)")
        
        ax6_twin = ax6.twinx()
        ax6_twin.plot(pupil_t.index, pupil_t.values, marker='s', color='#4682B4', linewidth=2, label="|ΔP| (Đồng tử)", linestyle='--')
        
        ax6.set_title("Chồng ghép Động lực học: Ánh nhìn và Đồng tử", fontweight="bold")
        ax6.set_xlabel("Time Bin")
        ax6.set_ylabel("Xác suất chuyển đổi (Prob_Transition)")
        ax6_twin.set_ylabel("Biến thiên đồng tử (|ΔP|)")
        ax6.set_xticks([1,2,3,4,5])
        
        # Legends
        lines1, labels1 = ax6.get_legend_handles_labels()
        lines2, labels2 = ax6_twin.get_legend_handles_labels()
        ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()
