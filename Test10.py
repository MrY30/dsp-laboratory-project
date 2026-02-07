import pyaudio
import numpy as np
import pydirectinput
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import sys

# --- CONFIGURATION ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
GAIN = 5.0

# CONTROLS
KEY_ACCEL = 'up'
KEY_BRAKE = 'down'
KEY_LEFT = 'left'
KEY_RIGHT = 'right'
KEY_RESPAWN = 'enter'

class AudioProcessor:
    def __init__(self, callback_update_ui):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        self.calibrating = False
        self.device_index = None
        self.callback_update_ui = callback_update_ui
        self.pressed = {'up':False, 'down':False, 'left':False, 'right':False, 'enter':False}
        
        # Thresholds (Default)
        self.silence_thresh = 500
        self.respawn_thresh = 15000
        self.pitch_thresh = 1000
        self.ratio_oe = 1.5 
        self.ratio_ssh = 3.0

    def get_devices(self):
        devices = []
        try:
            info = self.p.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            for i in range(0, numdevices):
                if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                    name = self.p.get_device_info_by_host_api_device_index(0, i).get('name')
                    devices.append((i, name))
        except Exception as e:
            print(f"Audio Device Error: {e}")
        return devices

    def start(self, device_index):
        if self.running: return
        self.device_index = device_index
        try:
            self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                      input_device_index=self.device_index, frames_per_buffer=CHUNK)
            self.running = True
            threading.Thread(target=self.process_loop, daemon=True).start()
        except Exception as e:
            print(f"Error starting stream: {e}")
            self.running = False

    def stop(self):
        self.running = False
        self.apply_keys(False, False, False, False)
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except: pass

    def get_spectrum(self):
        try:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float64) * GAIN
            
            rms = np.sqrt(np.mean(audio**2))
            window = np.hamming(len(audio))
            fft = np.fft.rfft(audio * window)
            mag = np.abs(fft)
            
            def get_band(low, high):
                idx_l = int(low/(RATE/CHUNK))
                idx_h = int(high/(RATE/CHUNK))
                return np.sum(mag[idx_l:idx_h])

            e_pitch = get_band(100, 300)
            e_low   = get_band(300, 800)
            e_mid   = get_band(2000, 4000)
            e_high  = get_band(5000, 10000)
            return rms, e_pitch, e_low, e_mid, e_high
        except:
            return 0,0,0,0,0

    def apply_keys(self, up, down, left, right):
        keys = [(KEY_ACCEL, up), (KEY_BRAKE, down), (KEY_LEFT, left), (KEY_RIGHT, right)]
        for k, s in keys:
            if s and not self.pressed[k]:
                pydirectinput.keyDown(k); self.pressed[k]=True
            elif not s and self.pressed[k]:
                pydirectinput.keyUp(k); self.pressed[k]=False

    def process_loop(self):
        while self.running:
            if self.calibrating:
                time.sleep(0.1)
                continue

            vol, e_pitch, e_low, e_mid, e_high = self.get_spectrum()
            up, down, left, right = False, False, False, False
            status_text = "Idle"
            
            if vol > self.respawn_thresh:
                pydirectinput.press(KEY_RESPAWN)
                status_text = "RESPAWN!"
            elif vol > self.silence_thresh:
                has_pitch = e_pitch > self.pitch_thresh
                if has_pitch:
                    ratio = e_mid / (e_low + 1)
                    if ratio > self.ratio_oe:
                        left = True; up = True; status_text = "LEFT (E)"
                    else:
                        right = True; up = True; status_text = "RIGHT (O)"
                else:
                    ratio = e_high / (e_mid + 1)
                    if ratio > self.ratio_ssh or ratio > 5.0:
                        up = True; status_text = "GAS (S)"
                    else:
                        down = True; status_text = "BRAKE (SH)"

            self.apply_keys(up, down, left, right)
            ui_data = {'vol': vol, 'pitch': e_pitch, 'status': status_text, 'keys': (up, down, left, right)}
            self.callback_update_ui(ui_data)

class VoiceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Voice Control Dashboard")
        self.geometry("500x600")
        self.configure(bg="#222222")
        self.resizable(False, False)
        self.processor = AudioProcessor(self.update_dashboard)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#222222", foreground="white")
        style.configure("Horizontal.TProgressbar", background="#00ff00", troughcolor="#444444")
        
        self.create_widgets()
        
    def create_widgets(self):
        header = tk.Frame(self, bg="#222222", pady=10)
        header.pack(fill="x")
        tk.Label(header, text="Mic:", bg="#222222", fg="#aaaaaa").pack(side="left", padx=10)
        
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(header, textvariable=self.device_var, width=40, state='readonly')
        self.device_combo.pack(side="left")
        
        devices = self.processor.get_devices()
        dev_list = [f"{d[0]}: {d[1]}" for d in devices]
        self.device_combo['values'] = dev_list
        if dev_list: self.device_combo.current(0)
        
        self.status_label = tk.Label(self, text="READY", font=("Arial", 24, "bold"), bg="#222222", fg="#00ff00")
        self.status_label.pack(pady=20)
        
        cross = tk.Frame(self, bg="#222222")
        cross.pack(pady=10)
        self.lbl_up = self.arrow(cross, "UP", 0, 1)
        self.lbl_left = self.arrow(cross, "LEFT", 1, 0)
        self.lbl_down = self.arrow(cross, "DOWN", 1, 1)
        self.lbl_right = self.arrow(cross, "RIGHT", 1, 2)
        
        self.lbl_up.grid(row=0, column=1, padx=5, pady=5)
        self.lbl_left.grid(row=1, column=0, padx=5, pady=5)
        self.lbl_down.grid(row=2, column=1, padx=5, pady=5)
        self.lbl_right.grid(row=1, column=2, padx=5, pady=5)
        
        met = tk.Frame(self, bg="#222222", padx=20, pady=20)
        met.pack(fill="x")
        tk.Label(met, text="Volume", bg="#222222", fg="white").pack(anchor="w")
        self.bar_vol = ttk.Progressbar(met, length=400)
        self.bar_vol.pack(fill="x", pady=(0, 10))
        tk.Label(met, text="Pitch", bg="#222222", fg="white").pack(anchor="w")
        self.bar_pitch = ttk.Progressbar(met, length=400)
        self.bar_pitch.pack(fill="x")
        
        btn = tk.Frame(self, bg="#222222", pady=20)
        btn.pack()
        self.btn_start = tk.Button(btn, text="START ENGINE", font=("Arial", 12, "bold"), bg="#00aa00", fg="white", command=self.toggle_start, width=15)
        self.btn_start.pack(side="left", padx=10)
        self.btn_calib = tk.Button(btn, text="CALIBRATE", font=("Arial", 12), bg="#444444", fg="white", command=self.run_calibration_wizard, width=15)
        self.btn_calib.pack(side="left", padx=10)

    def arrow(self, p, t, r, c):
        return tk.Label(p, text=t, width=10, height=3, bg="#333333", fg="#555555", font=("Arial", 10, "bold"))

    def toggle_start(self):
        if not self.processor.running:
            try:
                self.processor.start(int(self.device_combo.get().split(":")[0]))
                self.btn_start.config(text="STOP ENGINE", bg="#aa0000")
                self.device_combo.config(state="disabled")
                self.btn_calib.config(state="disabled")
            except: messagebox.showerror("Error", "Select Mic first")
        else:
            self.processor.stop()
            self.btn_start.config(text="START ENGINE", bg="#00aa00")
            self.device_combo.config(state="readonly")
            self.btn_calib.config(state="normal")
            self.reset_ui()

    def update_dashboard(self, data):
        self.after(0, lambda: self._safe_update(data))

    def _safe_update(self, data):
        self.bar_vol['value'] = min(100, (data['vol'] / 5000) * 100)
        self.bar_pitch['value'] = min(100, (data['pitch'] / 3000) * 100)
        self.status_label.config(text=data['status'])
        u, d, l, r = data['keys']
        self.lbl_up.config(bg="#00ff00" if u else "#333333", fg="black" if u else "#555555")
        self.lbl_down.config(bg="#ff0000" if d else "#333333", fg="white" if d else "#555555")
        self.lbl_left.config(bg="#00ff00" if l else "#333333", fg="black" if l else "#555555")
        self.lbl_right.config(bg="#00ff00" if r else "#333333", fg="black" if r else "#555555")

    def reset_ui(self):
        self.bar_vol['value'] = 0
        self.bar_pitch['value'] = 0
        self.status_label.config(text="STOPPED", fg="#aaaaaa")
        for l in [self.lbl_up, self.lbl_down, self.lbl_left, self.lbl_right]: l.config(bg="#333333")

    def run_calibration_wizard(self):
        try: idx = int(self.device_combo.get().split(":")[0])
        except: 
            messagebox.showerror("Error", "Select Mic first")
            return
        self.processor.start(idx)
        self.processor.calibrating = True
        
        steps = [("SILENCE", "Quiet..."), ("OOO", "Say 'OOO'"), ("EEE", "Say 'EEE'"), ("SHHH", "Say 'SHHH'"), ("SSSS", "Say 'SSSS'"), ("CLAP", "Clap!")]
        top = tk.Toplevel(self)
        top.geometry("400x300"); top.configure(bg="#333333")
        lbl = tk.Label(top, text="...", font=("Arial", 14), bg="#333333", fg="white"); lbl.pack(pady=40)
        pb = ttk.Progressbar(top, length=300); pb.pack(pady=20)
        results = {}

        def run_step(i):
            if i >= len(steps): finish(); return
            name, instr = steps[i]
            for c in range(3, 0, -1):
                lbl.config(text=f"{instr}\nRecording in {c}..."); top.update(); time.sleep(1)
            lbl.config(text="RECORDING...", fg="#00ff00"); top.update()
            
            data, st = [], time.time()
            while time.time()-st < 2.0:
                data.append(self.processor.get_spectrum())
                pb['value'] = ((time.time()-st)/2.0)*100; top.update()
            results[name] = data
            run_step(i+1)

        def finish():
            # Logic
            sil = max(np.mean([x[0] for x in results["SILENCE"]])*2, 500)
            do, de = [x for x in results["OOO"] if x[0]>sil], [x for x in results["EEE"] if x[0]>sil]
            po, pe = (np.median([x[1] for x in do]) if do else 1000), (np.median([x[1] for x in de]) if de else 1000)
            pt = min(po, pe) * 0.4
            ro = np.median([(x[3]/(x[2]+1)) for x in do]) if do else 0.5
            re = np.median([(x[3]/(x[2]+1)) for x in de]) if de else 2.0
            r_oe = (ro+re)/2
            r_sh = np.median([(x[4]/(x[3]+1)) for x in results["SHHH"]])
            r_s = np.median([(x[4]/(x[3]+1)) for x in results["SSSS"]])
            r_ssh = (r_sh+r_s)/2
            
            self.processor.silence_thresh = sil
            self.processor.pitch_thresh = pt
            self.processor.ratio_oe = r_oe
            self.processor.ratio_ssh = r_ssh
            self.processor.respawn_thresh = np.max([x[0] for x in results["CLAP"]])*0.8
            self.processor.calibrating = False; self.processor.stop()
            top.destroy()
            messagebox.showinfo("Done", "Calibration Complete!")

        self.after(100, lambda: run_step(0))

    def on_close(self):
        self.processor.stop()
        self.destroy()

if __name__ == "__main__":
    app = VoiceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()