import pyaudio
import numpy as np
import pydirectinput
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox

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
    """Handles the heavy lifting: Audio analysis and Key pressing in a separate thread."""
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
        info = self.p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = self.p.get_device_info_by_host_api_device_index(0, i).get('name')
                devices.append((i, name))
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
        # Release all keys
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
            
            # Helper to get energy in freq range
            def get_band(low, high):
                idx_l = int(low/(RATE/CHUNK))
                idx_h = int(high/(RATE/CHUNK))
                return np.sum(mag[idx_l:idx_h])

            # Bands
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
            
            # --- LOGIC GATES ---
            if vol > self.respawn_thresh:
                pydirectinput.press(KEY_RESPAWN)
                status_text = "RESPAWN!"
            
            elif vol > self.silence_thresh:
                # 1. Pitch Check
                has_pitch = e_pitch > self.pitch_thresh

                if has_pitch:
                    # Vowel (O/E)
                    ratio = e_mid / (e_low + 1)
                    if ratio > self.ratio_oe:
                        left = True; up = True
                        status_text = "LEFT (E)"
                    else:
                        right = True; up = True
                        status_text = "RIGHT (O)"
                else:
                    # Noise (S/SH)
                    ratio = e_high / (e_mid + 1)
                    if ratio > self.ratio_ssh or ratio > 5.0:
                        up = True
                        status_text = "GAS (S)"
                    else:
                        down = True
                        status_text = "BRAKE (SH)"

            self.apply_keys(up, down, left, right)
            
            # Send data back to UI (Volume, Pitch, Status, ActiveKeys)
            ui_data = {
                'vol': vol, 
                'pitch': e_pitch, 
                'status': status_text,
                'keys': (up, down, left, right)
            }
            self.callback_update_ui(ui_data)


class VoiceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Voice Control Dashboard")
        self.geometry("500x600")
        self.configure(bg="#222222")
        self.resizable(False, False)
        
        # Logic
        self.processor = AudioProcessor(self.update_dashboard)
        
        # Styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#222222", foreground="white")
        style.configure("TButton", background="#444444", foreground="white")
        
        self.create_widgets()
        
    def create_widgets(self):
        # 1. Header & Device Selection
        header_frame = tk.Frame(self, bg="#222222", pady=10)
        header_frame.pack(fill="x")
        
        tk.Label(header_frame, text="Microphone:", bg="#222222", fg="#aaaaaa").pack(side="left", padx=10)
        
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(header_frame, textvariable=self.device_var, width=40)
        self.device_combo['state'] = 'readonly'
        self.device_combo.pack(side="left")
        
        # Populate devices
        devices = self.processor.get_devices()
        dev_list = [f"{d[0]}: {d[1]}" for d in devices]
        self.device_combo['values'] = dev_list
        if dev_list: self.device_combo.current(0)
        
        # 2. Main Status Display
        self.status_label = tk.Label(self, text="READY", font=("Arial", 24, "bold"), bg="#222222", fg="#00ff00")
        self.status_label.pack(pady=20)
        
        # 3. Directional Indicators (The "Cross")
        cross_frame = tk.Frame(self, bg="#222222")
        cross_frame.pack(pady=10)
        
        self.lbl_up = self.make_arrow(cross_frame, "UP (S/E/O)", 0, 1)
        self.lbl_left = self.make_arrow(cross_frame, "LEFT (E)", 1, 0)
        self.lbl_down = self.make_arrow(cross_frame, "DOWN (SH)", 1, 1) # Center? No, let's put it below
        self.lbl_right = self.make_arrow(cross_frame, "RIGHT (O)", 1, 2)
        
        # Grid adjustments for specific layout:   UP
        #                                     LEFT  RIGHT
        #                                        DOWN
        self.lbl_up.grid(row=0, column=1, padx=5, pady=5)
        self.lbl_left.grid(row=1, column=0, padx=5, pady=5)
        self.lbl_down.grid(row=2, column=1, padx=5, pady=5)
        self.lbl_right.grid(row=1, column=2, padx=5, pady=5)
        
        # 4. Progress Bars (Metrics)
        metrics_frame = tk.Frame(self, bg="#222222", padx=20, pady=20)
        metrics_frame.pack(fill="x")
        
        tk.Label(metrics_frame, text="Volume", bg="#222222", fg="white").pack(anchor="w")
        self.bar_vol = ttk.Progressbar(metrics_frame, orient="horizontal", length=400, mode="determinate")
        self.bar_vol.pack(fill="x", pady=(0, 10))
        
        tk.Label(metrics_frame, text="Pitch (Vocal Cord)", bg="#222222", fg="white").pack(anchor="w")
        self.bar_pitch = ttk.Progressbar(metrics_frame, orient="horizontal", length=400, mode="determinate")
        self.bar_pitch.pack(fill="x")
        
        # 5. Buttons
        btn_frame = tk.Frame(self, bg="#222222", pady=20)
        btn_frame.pack()
        
        self.btn_start = tk.Button(btn_frame, text="START ENGINE", font=("Arial", 12, "bold"), 
                                   bg="#00aa00", fg="white", command=self.toggle_start, width=15)
        self.btn_start.pack(side="left", padx=10)
        
        self.btn_calib = tk.Button(btn_frame, text="CALIBRATE", font=("Arial", 12), 
                                   bg="#444444", fg="white", command=self.run_calibration_wizard, width=15)
        self.btn_calib.pack(side="left", padx=10)

    def make_arrow(self, parent, text, r, c):
        lbl = tk.Label(parent, text=text, width=12, height=3, bg="#333333", fg="#555555", font=("Arial", 10, "bold"))
        return lbl

    def toggle_start(self):
        if not self.processor.running:
            # Start
            try:
                sel = self.device_combo.get()
                idx = int(sel.split(":")[0])
                self.processor.start(idx)
                self.btn_start.config(text="STOP ENGINE", bg="#aa0000")
                self.device_combo.config(state="disabled")
                self.btn_calib.config(state="disabled")
            except:
                messagebox.showerror("Error", "Please select a microphone first.")
        else:
            # Stop
            self.processor.stop()
            self.btn_start.config(text="START ENGINE", bg="#00aa00")
            self.device_combo.config(state="readonly")
            self.btn_calib.config(state="normal")
            self.reset_ui()

    def update_dashboard(self, data):
        # Note: This is called from the thread. 
        # Tkinter isn't thread safe, but setting simple widget values often works.
        # Ideally, we queue this, but for this complexity, we'll direct set or use after.
        self.after(0, lambda: self._update_ui_safe(data))

    def _update_ui_safe(self, data):
        # 1. Bars
        # Scale volume roughly 0 to 5000
        vol_norm = min(100, (data['vol'] / 5000) * 100)
        self.bar_vol['value'] = vol_norm
        
        # Scale pitch roughly 0 to 3000
        pitch_norm = min(100, (data['pitch'] / 3000) * 100)
        self.bar_pitch['value'] = pitch_norm
        
        # 2. Text
        self.status_label.config(text=data['status'])
        
        # 3. Arrows
        u, d, l, r = data['keys']
        color_on = "#00ff00"
        color_off = "#333333"
        
        self.lbl_up.config(bg=color_on if u else color_off, fg="black" if u else "#555555")
        self.lbl_down.config(bg="#ff0000" if d else color_off, fg="white" if d else "#555555")
        self.lbl_left.config(bg=color_on if l else color_off, fg="black" if l else "#555555")
        self.lbl_right.config(bg=color_on if r else color_off, fg="black" if r else "#555555")

    def reset_ui(self):
        self.bar_vol['value'] = 0
        self.bar_pitch['value'] = 0
        self.status_label.config(text="STOPPED", fg="#aaaaaa")
        self.lbl_up.config(bg="#333333")
        self.lbl_down.config(bg="#333333")
        self.lbl_left.config(bg="#333333")
        self.lbl_right.config(bg="#333333")

    # --- CALIBRATION WIZARD ---
    def run_calibration_wizard(self):
        try:
            sel = self.device_combo.get()
            idx = int(sel.split(":")[0])
        except:
            messagebox.showerror("Error", "Select Mic first")
            return
            
        self.processor.start(idx)
        self.processor.calibrating = True
        
        # Define steps
        steps = [
            ("SILENCE", "Stay quiet to measure background noise."),
            ("OOO", "Say 'OOO' (Low Pitch)"),
            ("EEE", "Say 'EEE' (High Pitch)"),
            ("SHHH", "Say 'SHHH' (Brake noise)"),
            ("SSSS", "Say 'SSSS' (Gas noise)"),
            ("CLAP", "Clap loudly once (Respawn)")
        ]
        
        # Create Popup
        top = tk.Toplevel(self)
        top.title("Calibration")
        top.geometry("400x300")
        top.configure(bg="#333333")
        
        lbl_instr = tk.Label(top, text="Get Ready...", font=("Arial", 14), bg="#333333", fg="white", wraplength=350)
        lbl_instr.pack(pady=40)
        
        pb = ttk.Progressbar(top, length=300, mode="determinate")
        pb.pack(pady=20)
        
        results = {}

        def perform_step(step_index):
            if step_index >= len(steps):
                finish_calibration()
                return
            
            name, instruction = steps[step_index]
            lbl_instr.config(text=f"Step {step_index+1}/{len(steps)}: {name}\n\n{instruction}")
            top.update()
            
            # Countdown
            for i in range(3, 0, -1):
                lbl_instr.config(text=f"{instruction}\n\nRecording in {i}...")
                top.update()
                time.sleep(1)
            
            lbl_instr.config(text="RECORDING...", fg="#00ff00")
            top.update()
            
            # Record 2 seconds
            data_points = []
            st = time.time()
            while time.time() - st < 2.0:
                data_points.append(self.processor.get_spectrum())
                pb['value'] = ((time.time() - st) / 2.0) * 100
                top.update()
            
            results[name] = data_points
            perform_step(step_index + 1)

        def finish_calibration():
            # Apply Logic
            # 1. Silence
            silence_floor = max(np.mean([x[0] for x in results["SILENCE"]]) * 2.0, 500)
            
            # 2. Pitch Threshold (OOO vs EEE)
            data_o = [x for x in results["OOO"] if x[0] > silence_floor]
            data_e = [x for x in results["EEE"] if x[0] > silence_floor]
            
            p_o = np.median([x[1] for x in data_o]) if data_o else 1000
            p_e = np.median([x[1] for x in data_e]) if data_e else 1000
            pitch_thresh = min(p_o, p_e) * 0.4
            
            # 3. O/E Ratio
            r_o = np.median([(x[3]/(x[2]+1)) for x in data_o]) if data_o else 0.5
            r_e = np.median([(x[3]/(x[2]+1)) for x in data_e]) if data_e else 2.0
            ratio_oe = (r_o + r_e) / 2
            
            # 4. S/SH Ratio
            data_sh = results["SHHH"]
            data_s = results["SSSS"]
            r_sh = np.median([(x[4]/(x[3]+1)) for x in data_sh])
            r_s = np.median([(x[4]/(x[3]+1)) for x in data_s])
            ratio_ssh = (r_sh + r_s) / 2
            
            # 5. Respawn
            respawn = np.max([x[0] for x in results["CLAP"]]) * 0.8
            
            # Save to Processor
            self.processor.silence_thresh = silence_floor
            self.processor.pitch_thresh = pitch_thresh
            self.processor.ratio_oe = ratio_oe
            self.processor.ratio_ssh = ratio_ssh
            self.processor.respawn_thresh = respawn
            self.processor.calibrating = False
            self.processor.stop() # Stop temp stream
            
            top.destroy()
            messagebox.showinfo("Success", f"Calibration Complete!\n\nPitch Gate: {int(pitch_thresh)}\nO/E Ratio: {ratio_oe:.2f}\nS/SH Ratio: {ratio_ssh:.2f}")

        # Start the chain
        self.after(100, lambda: perform_step(0))

    def on_close(self):
        self.processor.stop()
        self.destroy()

if __name__ == "__main__":
    app = VoiceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()