import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import threading
import os
import datetime
from modules.data_processor import process_data
from modules.visualization import show_comparison_plots

# Nhập hàm vẽ riêng cho DATN1
from modules.visualization_datn1 import show_datn1_plots

class OQSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Phần mềm Đánh giá sự lưỡng lự (OQS & Classical)")
        self.geometry("750x650")
        
        self.files = []
        self.last_saved_file = None
        
        # Tiêu đề chung
        tk.Label(self, text="HỆ THỐNG PHÂN TÍCH CHUYỂN ĐỘNG MẮT", font=("Helvetica", 14, "bold")).pack(pady=5)
        
        # Tạo Notebook (Tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- TAB 1: PHÂN TÍCH PRO (Dashboard 6 Biểu đồ) ---
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="1. Hệ thống Phân tích Pro (Mới)")
        
        self.setup_tab1()
        
        # --- TAB 2: ĐỐI CHIẾU DATN1 (Tính theo Luận văn trước) ---
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="2. Đối chiếu Luận văn (DATN1)")
        
        self.setup_tab2()
        
        # --- TAB 3: BÁO CÁO CÔNG BỐ QUỐC TẾ (Q1 Journal Analytics) ---
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="3. Phân tích Lợi thế Lượng tử (Báo cáo Q1)")
        
        self.setup_tab3()
        
        # --- Chung: Progress & Log ---
        tk.Label(self, text="Tiến trình hệ thống:").pack(anchor=tk.W, padx=10)
        self.progress = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=700, mode='determinate')
        self.progress.pack(pady=5, padx=10)
        
        self.lbl_status = tk.Label(self, text="Trạng thái: Sẵn sàng")
        self.lbl_status.pack(pady=5)
        
        self.txt_log = tk.Text(self, height=10, width=85)
        self.txt_log.pack(pady=5, padx=10)
        
    def setup_tab1(self):
        tk.Label(self.tab1, text="Chế độ Pro: Tự động nội suy, làm mượt, và xuất 6 biểu đồ chuyên sâu.", fg="blue").pack(pady=10)
        
        self.btn_select = tk.Button(self.tab1, text="1. Chọn file dữ liệu thô (.xlsx, .csv)", command=self.select_files, width=45)
        self.btn_select.pack(pady=5)
        
        self.lbl_files = tk.Label(self.tab1, text="Chưa chọn file nào", fg="gray")
        self.lbl_files.pack(pady=5)
        
        self.btn_run = tk.Button(self.tab1, text="2. Bắt đầu phân tích (Lưu kết quả)", command=self.run_analysis, state=tk.DISABLED, width=45, bg="lightblue")
        self.btn_run.pack(pady=5)
        
        tk.Frame(self.tab1, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=50, pady=10)
        
        self.btn_load_res = tk.Button(self.tab1, text="3. Tải file kết quả (đã xử lý trước đó)", command=self.load_results, width=45)
        self.btn_load_res.pack(pady=5)
        
        self.btn_plot = tk.Button(self.tab1, text="4. Xem Báo cáo Pro (6 Biểu đồ & T-Test)", command=self.show_plots, state=tk.DISABLED, width=45, bg="lightgreen")
        self.btn_plot.pack(pady=5)

    def setup_tab2(self):
        tk.Label(self.tab2, text="Chế độ DATN1: Khớp mô hình nguyên bản và xuất 2 biểu đồ Động lực học.", fg="green").pack(pady=10)
        
        tk.Label(self.tab2, text="Lưu ý: Chế độ này dùng chung file kết quả đã xử lý để đảm bảo tính đồng nhất,\nnhưng sẽ vẽ biểu đồ và phân tích y hệt như Hình 4.2.1 và 4.2.5 trong Luận văn.", justify=tk.CENTER).pack(pady=5)
        
        self.btn_plot_datn1 = tk.Button(self.tab2, text="1. Xem Biểu đồ Luận văn (Hình 4.2.1 & 4.2.5)", command=self.show_datn1, state=tk.DISABLED, width=45, bg="lightyellow")
        self.btn_plot_datn1.pack(pady=20)

    def setup_tab3(self):
        tk.Label(self.tab3, text="PHÂN TÍCH CHUYÊN SÂU CẤP ĐỘ Q1 (NATURE / PSYCHOLOGICAL REVIEW)", font=("Helvetica", 11, "bold"), fg="darkred").pack(pady=10)
        
        tk.Label(self.tab3, text="Các thuật toán trích xuất 'Lợi thế Lượng tử':", justify=tk.CENTER).pack(pady=5)
        
        self.btn_zeno = tk.Button(self.tab3, text="1. Phân tích Hiệu ứng Zeno Lượng tử (Bootstrapping)", command=self.run_zeno_analysis, state=tk.NORMAL, width=65, bg="#ffcccb")
        self.btn_zeno.pack(pady=5)
        
        self.btn_tomography = tk.Button(self.tab3, text="2. Tái cấu trúc Ma trận Mật độ 3D (State Tomography)", command=self.run_tomography, state=tk.NORMAL, width=65, bg="#add8e6")
        self.btn_tomography.pack(pady=5)

        self.btn_clustering = tk.Button(self.tab3, text="3. Phân cụm Hồ sơ Nhận thức (K-Means Clustering)", command=self.run_clustering, state=tk.NORMAL, width=65, bg="#98fb98")
        self.btn_clustering.pack(pady=5)

        self.btn_qq = tk.Button(self.tab3, text="4. Nghiệm chứng Đẳng thức QQ (Quantum Question Equality)", command=self.run_qq_equality, state=tk.NORMAL, width=65, bg="#dda0dd")
        self.btn_qq.pack(pady=5)

        self.btn_qai = tk.Button(self.tab3, text="5. Dò tìm Chuyển pha Lượng tử (QAI Phase Transition)", command=self.run_qai, state=tk.NORMAL, width=65, bg="#ffeb3b")
        self.btn_qai.pack(pady=5)

        self.btn_nudge = tk.Button(self.tab3, text="6. Kỹ thuật Điều khiển: Mô phỏng 'Cú huých Lượng tử'", command=self.run_nudge, state=tk.NORMAL, width=65, bg="#ff7f50")
        self.btn_nudge.pack(pady=5)

        self.btn_hybrid = tk.Button(self.tab3, text="7. ĐỘT PHÁ CÔNG NGHỆ: Mạng Lai Lượng tử (Hybrid QNN)", command=self.run_hybrid, state=tk.NORMAL, width=65, bg="#8a2be2", font=("Helvetica", 10, "bold"), fg="white")
        self.btn_hybrid.pack(pady=5)
        
    def log(self, msg):
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)
        self.update_idletasks()
        
    def select_files(self):
        # Mặc định mở thẳng vào thư mục gốc của dự án thay vì C:\Users\...
        init_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        filepaths = filedialog.askopenfilenames(
            title="Chọn các file dữ liệu thô (.xlsx, .csv)",
            initialdir=init_dir,
            filetypes=(("Data files", "*.xlsx *.csv"), ("All files", "*.*"))
        )
        if filepaths:
            self.files = list(filepaths)
            self.lbl_files.config(text=f"Đã chọn {len(self.files)} file.")
            self.btn_run.config(state=tk.NORMAL)
            self.log(f"Đã nạp {len(self.files)} file.")

    def run_analysis(self):
        self.btn_run.config(state=tk.DISABLED)
        self.btn_select.config(state=tk.DISABLED)
        threading.Thread(target=self._process_thread, daemon=True).start()
        
    def _update_progress(self, current, total, msg):
        self.progress['value'] = (current / total) * 100
        self.lbl_status.config(text=msg)
        self.log(msg)
        
    def _process_thread(self):
        try:
            self.log("Bắt đầu xử lý...")
            df_prob, df_trans, df_fit, df_preprocessed = process_data(self.files, callback=self._update_progress)
            
            if df_fit is not None and not df_fit.empty:
                calc_dir = os.path.join(os.path.dirname(__file__), "Calculate")
                os.makedirs(calc_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                out_path = os.path.join(calc_dir, f"OQS_Results_{timestamp}.xlsx")
                with pd.ExcelWriter(out_path) as writer:
                    if df_prob is not None and not df_prob.empty:
                        df_prob.to_excel(writer, sheet_name="Prob_5bins", index=False)
                    if df_trans is not None and not df_trans.empty:
                        df_trans.to_excel(writer, sheet_name="Transitions_trial_5bins", index=False)
                    df_fit.to_excel(writer, sheet_name="OQS_Fit_Results", index=False)
                
                self.last_saved_file = out_path
                self.btn_plot.config(state=tk.NORMAL)
                self.btn_plot_datn1.config(state=tk.NORMAL)
                
                msg = f"Xử lý hoàn tất!\nKết quả lưu tại: {out_path}"
                self.log(msg)
                messagebox.showinfo("Thành công", msg)
            else:
                self.log("Không có dữ liệu hợp lệ.")
                
        except Exception as e:
            self.log(f"Lỗi: {e}")
            messagebox.showerror("Lỗi", str(e))
        finally:
            self.btn_run.config(state=tk.NORMAL)
            self.btn_select.config(state=tk.NORMAL)
            self.lbl_status.config(text="Trạng thái: Hoàn thành")

    def load_results(self):
        calc_dir = os.path.join(os.path.dirname(__file__), "Calculate")
        os.makedirs(calc_dir, exist_ok=True)
        path = filedialog.askopenfilename(
            title="Chọn file kết quả", 
            initialdir=calc_dir, 
            filetypes=[("Excel", "*.xlsx")]
        )
        if path:
            self.last_saved_file = path
            self.log(f"Đã tải file kết quả: {os.path.basename(path)}")
            self.btn_plot.config(state=tk.NORMAL)
            self.btn_plot_datn1.config(state=tk.NORMAL)
            messagebox.showinfo("Thành công", "Đã tải file kết quả.")

    def show_plots(self):
        if self.last_saved_file and os.path.exists(self.last_saved_file):
            show_comparison_plots(self.last_saved_file)
            
    def show_datn1(self):
        if self.last_saved_file and os.path.exists(self.last_saved_file):
            show_datn1_plots(self.last_saved_file)

    def run_zeno_analysis(self):
        try:
            import subprocess
            self.log("Đang chạy phân tích Hiệu ứng Zeno Lượng tử...")
            result = subprocess.run(["python", "zeno_effect_analysis.py"], cwd=os.path.dirname(__file__), capture_output=True, text=True, encoding='utf-8')
            
            output = result.stdout if result.stdout else result.stderr
            
            # Hiển thị trực tiếp lên GUI
            messagebox.showinfo("Kết quả Phân tích Zeno Lượng tử", output)
            self.log("Đã hoàn tất phân tích Zeno.")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def run_tomography(self):
        try:
            import subprocess, os
            self.log("Đang vẽ Tái cấu trúc Ma trận mật độ (3D Tomography)...")
            result = subprocess.run(["python", "q1_tomography.py"], cwd=os.path.dirname(__file__), capture_output=True, text=True, encoding='utf-8')
            output = result.stdout if result.stdout else result.stderr
            messagebox.showinfo("Hoàn tất Tomography & Entropy", output + "\n\nĐã xuất biểu đồ 3D Tomography thành công vào thư mục Figures!")
            fig_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Figures", "Fig4_QuantumTomography.png"))
            if os.path.exists(fig_path): os.startfile(fig_path)
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def run_clustering(self):
        try:
            import subprocess, os
            self.log("Đang chạy phân cụm K-Means Hồ sơ Nhận thức...")
            result = subprocess.run(["python", "clustering_analysis.py"], cwd=os.path.dirname(__file__), capture_output=True, text=True, encoding='utf-8')
            output = result.stdout if result.stdout else result.stderr
            messagebox.showinfo("Kết quả K-Means Clustering", output + "\n\nĐã phân cụm và xuất biểu đồ (Fig5) thành công!")
            fig_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Figures", "Fig5_CognitiveClusters.png"))
            if os.path.exists(fig_path): os.startfile(fig_path)
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def run_qq_equality(self):
        try:
            import subprocess, os
            self.log("Đang tính toán Đẳng thức Lượng tử (QQ Equality)...")
            result = subprocess.run(["python", "qq_equality_analysis.py"], cwd=os.path.dirname(__file__), capture_output=True, text=True, encoding='utf-8')
            output = result.stdout if result.stdout else result.stderr
            messagebox.showinfo("Kết quả Khảo nghiệm QQ Equality", output + "\n\nKhảo nghiệm QQ Equality hoàn tất! Biểu đồ (Fig6) đã xuất thành công!")
            fig_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Figures", "Fig6_QQEquality.png"))
            if os.path.exists(fig_path): os.startfile(fig_path)
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def run_qai(self):
        try:
            import subprocess, os
            self.log("Đang tính toán Chỉ số Lưỡng lự Lượng tử QAI...")
            result = subprocess.run(["python", "quantum_phase_transition.py"], cwd=os.path.dirname(__file__), capture_output=True, text=True, encoding='utf-8')
            output = result.stdout if result.stdout else result.stderr
            messagebox.showinfo("Phát hiện Chuyển pha Lượng tử", output + "\n\nĐã xuất biểu đồ Chuyển pha (Fig7) thành công!")
            fig_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Figures", "Fig7_PhaseTransition.png"))
            if os.path.exists(fig_path): os.startfile(fig_path)
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def run_nudge(self):
        try:
            import subprocess, os
            self.log("Đang chạy Mô phỏng Cú huých Lượng tử (Quantum Nudge)...")
            result = subprocess.run(["python", "q1_quantum_nudge.py"], cwd=os.path.dirname(__file__), capture_output=True, text=True, encoding='utf-8')
            output = result.stdout if result.stdout else result.stderr
            messagebox.showinfo("ĐỘT PHÁ TƯ TƯỞNG: Kỹ thuật Điều khiển Nhận thức", output + "\n\nĐã xuất biểu đồ (Fig8) thành công!")
            fig8_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Figures", "Fig8_QuantumNudge.png"))
            fig9_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Figures", "Fig9_NudgeTomography.png"))
            if os.path.exists(fig8_path): os.startfile(fig8_path)
            if os.path.exists(fig9_path): os.startfile(fig9_path)
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def run_hybrid(self):
        try:
            import subprocess, os
            self.log("Đang chạy Mạng nơ-ron Lai Lượng tử (Hybrid QNN)...")
            result = subprocess.run(["python", "q1_hybrid_qnn.py"], cwd=os.path.dirname(__file__), capture_output=True, text=True, encoding='utf-8')
            output = result.stdout if result.stdout else result.stderr
            messagebox.showinfo("Đỉnh cao AI: Hybrid Quantum Neural Network", output + "\n\nĐã xuất biểu đồ (Fig10) thành công!")
            fig10_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Figures", "Fig10_HybridQNN.png"))
            if os.path.exists(fig10_path): os.startfile(fig10_path)
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def run_q1_nature(self):
        try:
            import subprocess, os
            self.log("Đang chạy Xuất Bức tranh Composite chuẩn báo Q1...")
            result = subprocess.run(["python", "q1_nature_paper_figures.py"], cwd=os.path.dirname(__file__), capture_output=True, text=True, encoding='utf-8')
            output = result.stdout if result.stdout else result.stderr
            messagebox.showinfo("Xuất Bức tranh Composite chuẩn báo Q1", output + "\n\nĐã xuất biểu đồ thành công!")
            fig_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Figures", "Fig_Q1_Composite.png"))
            if os.path.exists(fig_path): os.startfile(fig_path)
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

if __name__ == "__main__":
    app = OQSApp()
    app.mainloop()
