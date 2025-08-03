import os
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pywt
from scipy.signal import find_peaks, butter, filtfilt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Hàm vẽ tín hiệu ECG
def plot_ecg(file_path):
    try:
        df = pd.read_csv(file_path, header=None, skiprows=2500, encoding='latin1')
        raw_adc = df[0].values
        ecg_voltage = raw_adc * (3.3 / 4095)

        # Lọc wavelet db4 cấp 4
        level = 4
        coeffs = pywt.wavedec(ecg_voltage, 'db4', level=level)
        #coeffs[0] = np.zeros_like(coeffs[0])  # A4: 0–15.625 Hz
        coeffs[1] = np.zeros_like(coeffs[1])  # D4: 15.625–31.25 Hz
        coeffs[2] = np.zeros_like(coeffs[2])  # D3: 31.25–62.5 Hz
        coeffs[3] = np.zeros_like(coeffs[3])  # D2: 62.5–125 Hz
        coeffs[4] = np.zeros_like(coeffs[4])  # D1: 125–250 Hz
        ecg_filtered = pywt.waverec(coeffs, 'db4')

        # Đồng bộ độ dài
        if len(ecg_filtered) > len(ecg_voltage):
            ecg_filtered = ecg_filtered[:len(ecg_voltage)]
        elif len(ecg_filtered) < len(ecg_voltage):
            ecg_filtered = np.pad(ecg_filtered, (0, len(ecg_voltage) - len(ecg_filtered)), mode='edge')

        fs = 500
        time = np.linspace(0, len(ecg_voltage) / fs, len(ecg_voltage))

        # Tìm đỉnh R
        distance = int(0.6 * fs)
        peaks, _ = find_peaks(ecg_filtered, distance=distance, height = np.mean(ecg_filtered) + 0.3 * (np.max(ecg_filtered) - np.mean(ecg_filtered)))
        rr_intervals = np.diff(peaks) / fs
        bpm = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
        beat_count = len(peaks)

        # Giao diện Tkinter
        win = tk.Toplevel()
        win.title("Đồ thị ECG")

        # Vẽ đồ thị
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(time, ecg_filtered, label='ECG đã lọc', color='blue')
        ax.plot(time[peaks], ecg_filtered[peaks], 'ro', label='Đỉnh R')
        ax.set_xlabel('Thời gian (s)')
        ax.set_ylabel('Điện áp (V)')
        ax.set_title('Tín hiệu ECG')
        # Thêm nhịp tim vào dưới figure (trong ảnh)
        fig.subplots_adjust(bottom=0.2)  # chừa chỗ dưới
        fig.text(0.5, 0.05, f"Số nhịp: {beat_count}   |   Nhịp tim trung bình: {bpm:.1f} BPM",
        ha='center', fontsize=12, bbox=dict(facecolor='lightyellow', edgecolor='black', boxstyle='round,pad=0.5'))

        ax.grid(True)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Thanh công cụ matplotlib
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
    except Exception as e:
        messagebox.showerror("Lỗi ECG", str(e))

# Hàm vẽ tín hiệu PPG
def plot_ppg(file_path):
    try:
        df = pd.read_csv(file_path, header=None, skiprows=500, encoding='latin1')
        red = df[0].values * (3.3 / 16384)
        ir = df[1].values * (3.3 / 16384)

        def wavelet_filter(sig):
            coeffs = pywt.wavedec(sig, 'coif5', level=4)
            #coeffs[0] = np.zeros_like(coeffs[0])  # A4: 0–3.125 Hz   ← Thành phần chứa nhịp tim chính (~1Hz)
            coeffs[1] = np.zeros_like(coeffs[1])  # D4: 3.125–6.25 Hz
            coeffs[2] = np.zeros_like(coeffs[2])  # D3: 6.25–12.5 Hz
            coeffs[3] = np.zeros_like(coeffs[3])  # D2: 12.5–25 Hz
            coeffs[4] = np.zeros_like(coeffs[4])  # D1: 25–50 Hz     ← Thành phần nhiễu cao nhất (EMG, nhiễu điện)
            return pywt.waverec(coeffs, 'coif5')

        #red_f = wavelet_filter(red)
        #ir_f = wavelet_filter(ir)
        red_f = red
        ir_f = ir
        # Đồng bộ độ dài sau lọc
        if len(red_f) > len(red):
            red_f = red_f[:len(red)]
        elif len(red_f) < len(red):
            red_f = np.pad(red_f, (0, len(red) - len(red_f)), mode='edge')

        if len(ir_f) > len(ir):
            ir_f = ir_f[:len(ir)]
        elif len(ir_f) < len(ir):
            ir_f = np.pad(ir_f, (0, len(ir) - len(ir_f)), mode='edge')

        spo2 = []
        for i in range(0, len(red_f)-100, 100):
            r, i_ = red_f[i:i+100], ir_f[i:i+100]
            R = (np.std(r)/np.mean(r)) / (np.std(i_)/np.mean(i_))
            spo2.append(110 - 25 * R)

        time = np.linspace(0, len(red_f)/100, len(red_f))

        win = tk.Toplevel()
        win.title("Đồ thị PPG")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        peaks_red = find_peaks(red_f, distance=60)[0]
        ax1.plot(time, red_f, color='red', label='PPG Red')
        ax1.plot(time[peaks_red], red_f[peaks_red], 'ro')
        ax1.legend()
        ax1.grid(True)

        peaks_ir = find_peaks(ir_f, distance=60)[0]
        ax2.plot(time, ir_f, color='blue', label='PPG IR')
        ax2.plot(time[peaks_ir], ir_f[peaks_ir], 'bo')
        ax2.set_title(f"SpO₂ trung bình: {np.mean(spo2):.2f}%")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    except Exception as e:
        messagebox.showerror("Lỗi PPG", str(e))


# Hàm vẽ tín hiệu PCG
def plot_pcg(file_path):
    try:
        df = pd.read_csv(file_path, header=None, skiprows=7500, encoding='latin1')
        pcg = (df[0].values * 3.3 / 16384)

        # Lọc wavelet db4 cấp 6
        level = 6
        coeffs = pywt.wavedec(pcg, 'db4', level=level)

        # Gán bằng 0 các phần nằm ngoài 23.44 – 93.75 Hz
        coeffs[0] = np.zeros_like(coeffs[0])  # A6: 0–11.72 Hz
        coeffs[1] = np.zeros_like(coeffs[1])  # D6: 11.72–23.44 Hz
        coeffs[2] = np.zeros_like(coeffs[2])  # D5: 23.44–46.88 Hz
        coeffs[3] = np.zeros_like(coeffs[3])  # D4: 46.88–93.75 Hz
        #coeffs[4] = np.zeros_like(coeffs[4])  # D3: 93.75–187.5 Hz
        #coeffs[5] = np.zeros_like(coeffs[5])  # D2: 187.5–375 Hz
        #coeffs[6] = np.zeros_like(coeffs[6])  # D1: 375–750 Hz
        filtered = pywt.waverec(coeffs, 'db4')
        #filtered = pcg
   

        # Đồng bộ độ dài sau lọc
        if len(filtered) > len(pcg):
            filtered = filtered[:len(pcg)]
        elif len(filtered) < len(pcg):
            filtered = np.pad(filtered, (0, len(pcg) - len(filtered)), mode='edge')

        fs = 1500
        time = np.linspace(0, len(filtered) / fs, len(filtered))

        # Vẽ đồ thị
        win = tk.Toplevel()
        win.title("Đồ thị PCG")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(time, filtered, label='PCG đã lọc', color='blue')
        ax.set_xlabel("Thời gian (s)")
        ax.set_ylabel("Biên độ")
        ax.set_title("Tín hiệu PCG")
        ax.grid(True)
        ax.legend()

        # Hiển thị GUI
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    except Exception as e:
        messagebox.showerror("Lỗi PCG", str(e))


# Giao diện GUI Tkinter
class SignalAnalyzerApp:
    def __init__(self, root):
        #Giao diện chung
        root.resizable(False, False)
        self.root = root
        root.title("Hiển thị tín hiệu tim mạch")
        root.configure(bg="white")
        root.geometry("500x300")
        
        #biến lấy đường dẫn
        self.ecg_path = tk.StringVar()
        self.ppg_path = tk.StringVar()
        self.pcg_path = tk.StringVar()
        
        #thông báo lỗi
        self.error_label = tk.Label(root, text="", fg="red", bg="white")
        self.error_label.pack(pady=(5,10))
        
        #3 vùng lấy thông tin đường dẫn
        self.create_file_input("Chọn file ECG:", self.ecg_path)
        self.create_file_input("Chọn file PPG:", self.ppg_path)
        self.create_file_input("Chọn file PCG:", self.pcg_path)
        
        #giao diện các nút
        btn_frame = tk.Frame(root, bg="white")
        btn_frame.pack(pady=15) #khoảng cách trên dưới
        tk.Button(btn_frame, text="Vẽ ECG", command=self.draw_ecg, width=10).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Vẽ PPG", command=self.draw_ppg, width=10).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="Vẽ PCG", command=self.draw_pcg, width=10).grid(row=0, column=2, padx=5)
        tk.Button(btn_frame, text="Thoát", command=root.quit, width=10).grid(row=0, column=3, padx=5)

    def create_file_input(self, label_text, var):
        frame = tk.Frame(self.root, bg="white")
        frame.pack(fill='x', padx=10, pady=3)

        label = tk.Label(frame, text=label_text, width=15, anchor='w', bg="white")
        label.pack(side='left')

        entry = tk.Entry(frame, textvariable=var, width=40)
        entry.pack(side='left', padx=(0,5))

        btn_browse = tk.Button(frame, text="Chọn file", command=lambda: self.browse_file(var))
        btn_browse.pack(side='left')

    def browse_file(self, var):
        filename = filedialog.askopenfilename()
        if filename:
            var.set(filename)
            self.error_label.config(text="")

    def check_file(self, path):
        if not path or not os.path.exists(path):
            self.error_label.config(text="Đường dẫn file không hợp lệ. Vui lòng chọn lại.")
            return False
        self.error_label.config(text="")
        return True

    def draw_ecg(self):
        if self.check_file(self.ecg_path.get()):
            plot_ecg(self.ecg_path.get())

    def draw_ppg(self):
        if self.check_file(self.ppg_path.get()):
            plot_ppg(self.ppg_path.get())

    def draw_pcg(self):
        if self.check_file(self.pcg_path.get()):
            plot_pcg(self.pcg_path.get())

if __name__ == "__main__":
    root = tk.Tk()
    app = SignalAnalyzerApp(root)
    root.mainloop()
