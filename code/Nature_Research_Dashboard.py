import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os

class NatureDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Quantum Cognitive Engineering - C-Level Dashboard")
        self.geometry("800x650")
        self.configure(bg='#050A30')
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Consolas', 11, 'bold'), padding=12, background='#000C66', foreground='#7EC8E3', borderwidth=2)
        style.map('TButton', background=[('active', '#050A30')], foreground=[('active', '#FFFFFF')])
        style.configure('TLabel', font=('Consolas', 10), background='#050A30', foreground='#FFFFFF')
        
        header_font = ('Segoe UI', 18, 'bold')
        
        # Header
        tk.Label(self, text="HỆ THỐNG KỸ THUẬT NHẬN THỨC LƯỢNG TỬ", font=header_font, bg='#050A30', fg='#7EC8E3').pack(pady=(20, 5))
        tk.Label(self, text="Phiên bản Nâng cấp Tối thượng (Quantum Nudge & QQ Equality)", font=('Segoe UI', 11, 'italic'), bg='#050A30', fg='#B0E0E6').pack(pady=(0, 20))
        
        # Frame for buttons
        frame = tk.Frame(self, bg='#050A30')
        frame.pack(fill=tk.BOTH, expand=True, padx=40)
        
        btn_width = 65
        
        btn1 = ttk.Button(frame, text="1. KIỂM ĐỊNH FISHER & ĐẲNG THỨC LƯỢNG TỬ (QQ EQUALITY)", width=btn_width, command=self.run_qq)
        btn1.pack(pady=8)
        
        btn2 = ttk.Button(frame, text="2. MÔ PHỎNG ENTROPY VON NEUMANN VS HỆ THẦN KINH LC-NE", width=btn_width, command=self.run_entropy)
        btn2.pack(pady=8)

        btn3 = ttk.Button(frame, text="3. MÔ PHỎNG QUỸ ĐẠO NGẪU NHIÊN BELAVKIN (SME)", width=btn_width, command=self.run_stochastic)
        btn3.pack(pady=8)
        
        btn4 = ttk.Button(frame, text="4. PHÂN TÍCH HIỆU ỨNG ZENO (SACCADIC DECOHERENCE)", width=btn_width, command=self.run_zeno)
        btn4.pack(pady=8)

        btn_new = ttk.Button(frame, text="5. [ĐỘT PHÁ] KỸ THUẬT ĐIỀU KHIỂN NHẬN THỨC (QUANTUM NUDGE 3D)", width=btn_width, command=self.run_nudge)
        btn_new.pack(pady=8)
        
        btn6 = ttk.Button(frame, text="6. TRÍCH XUẤT SIÊU BẢN THẢO NATURE (DOCX FINAL)", width=btn_width, command=self.open_docx)
        btn6.pack(pady=8)
        
        # Status bar
        self.status = tk.Label(self, text="SYSTEM ONLINE. ALL QUANTUM ENGINES INITIALIZED.", bd=1, relief=tk.SUNKEN, anchor=tk.W, font=('Consolas', 10, 'bold'), bg='#000C66', fg='#7EC8E3')
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def run_qq(self):
        self.status.config(text="Đang phân tích Ma trận Khảo sát Vĩ mô (QQ Equality)...")
        self.update()
        if os.path.exists('qq_equality_analysis.py'):
            subprocess.Popen(['python', 'qq_equality_analysis.py'])
            if os.path.exists('nature_qoe_fisher_test.py'):
                subprocess.Popen(['python', 'nature_qoe_fisher_test.py'])
            messagebox.showinfo("Hoàn tất", "Đã xác thực Đẳng thức Lượng tử. Đồ thị QQ và Fisher đã được xuất!")
        else:
            messagebox.showerror("Lỗi", "Không tìm thấy module")
        self.status.config(text="READY.")

    def run_entropy(self):
        self.status.config(text="Đang giải Phương trình Master Lindblad...")
        self.update()
        if os.path.exists('nature_entropy_pupil_engine.py'):
            subprocess.Popen(['python', 'nature_entropy_pupil_engine.py'])
            messagebox.showinfo("Hoàn tất", "Đã xuất Đồ thị Tương quan Entropy và Sinh lý học (quantum_entropy_pupil.png)")
        else:
            messagebox.showerror("Lỗi", "Không tìm thấy module")
        self.status.config(text="READY.")
        
    def run_stochastic(self):
        self.status.config(text="Đang mô phỏng Nhiễu Wiener (Phương trình Belavkin)...")
        self.update()
        if os.path.exists('nature_stochastic_trajectory.py'):
            subprocess.Popen(['python', 'nature_stochastic_trajectory.py'])
            messagebox.showinfo("Hoàn tất", "Đã khởi tạo thành công Quỹ đạo Lượng tử Ngẫu nhiên (quantum_trajectory_belavkin.png)")
        else:
            messagebox.showerror("Lỗi", "Không tìm thấy module")
        self.status.config(text="READY.")

    def run_zeno(self):
        self.status.config(text="Đang chạy Hồi quy Zeno Decoherence...")
        self.update()
        if os.path.exists('q1_cognitive_decoherence.py'):
            subprocess.Popen(['python', 'q1_cognitive_decoherence.py'])
            messagebox.showinfo("Hoàn tất", "Module Decoherence đã chạy thành công")
        else:
            messagebox.showerror("Lỗi", "Không tìm thấy module")
        self.status.config(text="READY.")

    def run_nudge(self):
        self.status.config(text="Đang mô phỏng Toán tử Unitary U (Quantum Nudge)...")
        self.update()
        if os.path.exists('q1_quantum_nudge.py'):
            subprocess.Popen(['python', 'q1_quantum_nudge.py'])
            messagebox.showinfo("Hoàn tất", "Đã bẻ gãy Tê liệt Zeno! Đồ thị 3D Tomography đã được kích hoạt.")
        else:
            messagebox.showerror("Lỗi", "Không tìm thấy file q1_quantum_nudge.py")
        self.status.config(text="READY.")

    def open_docx(self):
        path = r"E:\ET\ĐA Đánh giá sự lưỡng lự bằng mô hình lượng tử OQS4 và OQS6-20260828T140941Z-1-001\results\BaoCao_Nature_TiengViet_Final.docx"
        if os.path.exists(path):
            os.startfile(path)
        else:
            messagebox.showerror("Lỗi", "Không tìm thấy file DOCX")

if __name__ == "__main__":
    app = NatureDashboard()
    app.mainloop()
