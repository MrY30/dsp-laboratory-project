import pyaudio
import numpy as np
import pydirectinput
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque, Counter

# --- CONFIGURATION ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
GAIN = 3.0  # Software gain to boost mic sensitivity

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
        
        # State tracking
        self.pressed = {'up':False, 'down':False, 'left':False, 'right':False, 'enter':False}
        self.command_buffer = deque(maxlen=5) # Smoothing buffer (Stores last 5 commands)
        
        # Thresholds (Will be overwritten by Calibration)
        self.thresh_silence = 300    # RMS Volume
        self.thresh_zcr = 0.15       # Split between Vowel (Steer) and Fricative (Pedal)
        self.thresh_vowel_cent = 1500 # Split between OOO (Left) and EEE (Right)
        self.thresh_fric_cent = 4000  # Split between SHHH (Brake) and SSSS (Gas)
        self.thresh_respawn = 20000   # Volume for Clap/Respawn

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
        self.apply_keys("IDLE") # Release all keys
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except: pass

    def get_features(self):
        """
        Extracts robust DSP features: RMS, Zero Crossing Rate, and Spectral Centroid.
        """
        try:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            # Convert to float array for processing
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float64) * GAIN
            
            # 1. RMS (Volume)
            rms = np.sqrt(np.mean(audio**2))
            
            # 2. Zero Crossing Rate (ZCR) - Good for distinguishing Noise vs Tone
            # Count sign changes normalized by frame length
            zcr = ((audio[:-1] * audio[1:]) < 0).sum() / len(audio)
            
            # 3. Spectral Centroid - "Brightness" of the sound
            # Frequency Domain Analysis
            window = np.hamming(len(audio))
            fft_mag = np.abs(np.fft.rfft(audio * window))
            freqs = np.fft.rfftfreq(len(audio), 1/RATE)
            
            # Weighted average of frequencies
            sum_mag = np.sum(fft_mag)
            if sum_mag < 1e-9: centroid = 0
            else: centroid = np.sum(freqs * fft_mag) / sum_mag

            return rms, zcr, centroid
        except:
            return 0, 0, 0

    def decide_command(self, rms, zcr, centroid):
        """
        Decision Tree Logic based on DSP features
        """
        if rms > self.thresh_respawn:
            return "RESPAWN"
        
        if rms < self.thresh_silence:
            return "IDLE"

        # Step 1: Voiced (Steer) vs Unvoiced (Pedal) using ZCR
        if zcr < self.thresh_zcr:
            # Low ZCR = Vowel Sounds (Steering)
            # Step 2a: OOO vs EEE using Centroid
            if centroid < self.thresh_vowel_cent:
                return "LEFT"  # OOO (Dark sound)
            else:
                return "RIGHT" # EEE (Bright sound)
        else:
            # High ZCR = Fricative Sounds (Pedals)
            # Step 2b: SHHH vs SSSS using Centroid
            if centroid < self.thresh_fric_cent:
                return "BRAKE" # SHHH (Lower noise)
            else:
                return "GAS"   # SSSS (Higher noise)

    def apply_keys(self, command):
        # Map commands to boolean states
        target = {'up':False, 'down':False, 'left':False, 'right':False, 'enter':False}
        
        if command == "GAS": target['up'] = True
        elif command == "BRAKE": target['down'] = True
        elif command == "LEFT": target['left'] = True
        elif command == "RIGHT": target['right'] = True
        elif command == "RESPAWN": target['enter'] = True
        elif command == "IDLE": pass

        # Apply to pydirectinput only if state changed
        for k, pressed_now in target.items():
            if pressed_now and not self.pressed[k]:
                pydirectinput.keyDown(k)
                self.pressed[k] = True
            elif not pressed_now and self.pressed[k]:
                pydirectinput.keyUp(k)
                self.pressed[k] = False

    def process_loop(self):
        while self.running:
            if self.calibrating:
                time.sleep(0.1)
                continue

            rms, zcr, centroid = self.get_features()
            
            # Raw Decision
            raw_cmd = self.decide_command(rms, zcr, centroid)
            
            # Smoothing (Majority Vote)
            self.command_buffer.append(raw_cmd)
            # If buffer isn't full yet, just use raw
            if len(self.command_buffer) < 3:
                final_cmd = raw_cmd
            else:
                # Get the most common command in the last 5 frames
                final_cmd = Counter(self.command_buffer).most_common(1)[0][0]

            self.apply_keys(final_cmd)
            
            # UI Update Data
            keys_tuple = (self.pressed['up'], self.pressed['down'], self.pressed['left'], self.pressed['right'])
            ui_data = {
                'vol': rms, 
                'zcr': zcr, 
                'cent': centroid, 
                'status': final_cmd, 
                'keys': keys_tuple
            }
            self.callback_update_ui(ui_data)

class VoiceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Voice Control Dashboard V2 (DSP Enhanced)")
        self.geometry("600x650")
        self.configure(bg="#222222")
        self.resizable(False, False)
        self.processor = AudioProcessor(self.update_dashboard)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#222222", foreground="white")
        style.configure("Horizontal.TProgressbar", background="#00ff00", troughcolor="#444444")
        
        self.create_widgets()
        
    def create_widgets(self):
        # --- HEADER ---
        header = tk.Frame(self, bg="#222222", pady=10)
        header.pack(fill="x")
        tk.Label(header, text="Mic Source:", bg="#222222", fg="#aaaaaa").pack(side="left", padx=10)
        
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(header, textvariable=self.device_var, width=50, state='readonly')
        self.device_combo.pack(side="left")
        
        devices = self.processor.get_devices()
        dev_list = [f"{d[0]}: {d[1]}" for d in devices]
        self.device_combo['values'] = dev_list
        if dev_list: self.device_combo.current(0)
        
        # --- STATUS ---
        self.status_label = tk.Label(self, text="READY", font=("Arial", 32, "bold"), bg="#222222", fg="#00ff00")
        self.status_label.pack(pady=10)
        
        # --- ARROW GRID ---
        cross = tk.Frame(self, bg="#222222")
        cross.pack(pady=10)
        self.lbl_up = self.arrow(cross, "GAS (S)", 0, 1)
        self.lbl_left = self.arrow(cross, "LEFT (O)", 1, 0)
        self.lbl_down = self.arrow(cross, "BRAKE (SH)", 1, 1)
        self.lbl_right = self.arrow(cross, "RIGHT (E)", 1, 2)
        
        self.lbl_up.grid(row=0, column=1, padx=5, pady=5)
        self.lbl_left.grid(row=1, column=0, padx=5, pady=5)
        self.lbl_down.grid(row=2, column=1, padx=5, pady=5)
        self.lbl_right.grid(row=1, column=2, padx=5, pady=5)
        
        # --- METRICS ---
        met = tk.Frame(self, bg="#222222", padx=20, pady=20)
        met.pack(fill="x")
        
        tk.Label(met, text="Volume (RMS)", bg="#222222", fg="white", font=("Arial", 8)).pack(anchor="w")
        self.bar_vol = ttk.Progressbar(met, length=550)
        self.bar_vol.pack(fill="x", pady=(0, 5))
        
        tk.Label(met, text="Zero Crossing Rate (ZCR) - [Vowel vs Noise]", bg="#222222", fg="white", font=("Arial", 8)).pack(anchor="w")
        self.bar_zcr = ttk.Progressbar(met, length=550)
        self.bar_zcr.pack(fill="x", pady=(0, 5))

        tk.Label(met, text="Spectral Centroid - [Tone Brightness]", bg="#222222", fg="white", font=("Arial", 8)).pack(anchor="w")
        self.bar_cent = ttk.Progressbar(met, length=550)
        self.bar_cent.pack(fill="x", pady=(0, 5))
        
        # --- BUTTONS ---
        btn = tk.Frame(self, bg="#222222", pady=20)
        btn.pack()
        self.btn_start = tk.Button(btn, text="START ENGINE", font=("Arial", 12, "bold"), bg="#00aa00", fg="white", command=self.toggle_start, width=15)
        self.btn_start.pack(side="left", padx=10)
        self.btn_calib = tk.Button(btn, text="CALIBRATE", font=("Arial", 12), bg="#444444", fg="white", command=self.run_calibration_wizard, width=15)
        self.btn_calib.pack(side="left", padx=10)

    def arrow(self, p, t, r, c):
        return tk.Label(p, text=t, width=12, height=3, bg="#333333", fg="#555555", font=("Arial", 10, "bold"))

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
        # Update Bars with scaling
        self.bar_vol['value'] = min(100, (data['vol'] / 5000) * 100)
        self.bar_zcr['value'] = min(100, (data['zcr'] / 0.5) * 100)       # ZCR usually 0.0 to 0.5
        self.bar_cent['value'] = min(100, (data['cent'] / 8000) * 100)    # Centroid usually 0 to 8000Hz
        
        self.status_label.config(text=data['status'])
        
        # Update Keys visual
        u, d, l, r = data['keys']
        self.lbl_up.config(bg="#00ff00" if u else "#333333", fg="black" if u else "#555555")
        self.lbl_down.config(bg="#ff0000" if d else "#333333", fg="white" if d else "#555555")
        self.lbl_left.config(bg="#00ff00" if l else "#333333", fg="black" if l else "#555555")
        self.lbl_right.config(bg="#00ff00" if r else "#333333", fg="black" if r else "#555555")

    def reset_ui(self):
        self.bar_vol['value'] = 0
        self.bar_zcr['value'] = 0
        self.bar_cent['value'] = 0
        self.status_label.config(text="STOPPED", fg="#aaaaaa")
        for l in [self.lbl_up, self.lbl_down, self.lbl_left, self.lbl_right]: l.config(bg="#333333")

    def run_calibration_wizard(self):
        try: idx = int(self.device_combo.get().split(":")[0])
        except: 
            messagebox.showerror("Error", "Select Mic first")
            return
        
        self.processor.start(idx)
        self.processor.calibrating = True
        
        # Calibration Steps
        steps = [
            ("SILENCE", "Stay Quiet\n(Background Noise Level)"), 
            ("OOO", "Say 'OOO'\n(Left Turn - Low Pitch)"), 
            ("EEE", "Say 'EEE'\n(Right Turn - High Pitch)"), 
            ("SHHH", "Say 'SHHH'\n(Brake - Soft Noise)"), 
            ("SSSS", "Say 'SSSS'\n(Gas - Sharp Noise)"),
            ("CLAP", "CLAP LOUD!\n(Respawn Trigger)")
        ]
        
        top = tk.Toplevel(self)
        top.geometry("450x350"); top.configure(bg="#333333")
        lbl = tk.Label(top, text="...", font=("Arial", 14), bg="#333333", fg="white"); lbl.pack(pady=40)
        pb = ttk.Progressbar(top, length=350); pb.pack(pady=20)
        results = {}

        def run_step(i):
            if i >= len(steps): finish(); return
            name, instr = steps[i]
            
            # Countdown
            for c in range(3, 0, -1):
                lbl.config(text=f"{instr}\nRecording in {c}..."); top.update(); time.sleep(1)
            
            lbl.config(text="RECORDING...", fg="#00ff00"); top.update()
            
            # Record Data
            data_points = []
            st = time.time()
            while time.time()-st < 2.0:
                # Capture (RMS, ZCR, Centroid)
                data_points.append(self.processor.get_features())
                pb['value'] = ((time.time()-st)/2.0)*100; top.update()
                time.sleep(0.05) # slight delay to not spam
            
            results[name] = data_points
            run_step(i+1)

        def finish():
            # --- INTELLIGENT THRESHOLD CALCULATION ---
            
            # 1. Silence Threshold (Max RMS detected during silence * 2)
            silence_rms = np.max([x[0] for x in results["SILENCE"]])
            self.processor.thresh_silence = max(silence_rms * 2, 300)

            # 2. Respawn Threshold (80% of clap volume)
            clap_rms = np.max([x[0] for x in results["CLAP"]])
            self.processor.thresh_respawn = clap_rms * 0.8

            # 3. ZCR Threshold (Split Vowels vs Fricatives)
            # Average ZCR of Vowels (OOO, EEE) vs Fricatives (SHHH, SSSS)
            zcr_vowels = np.mean([x[1] for x in results["OOO"] + results["EEE"]])
            zcr_frics = np.mean([x[1] for x in results["SHHH"] + results["SSSS"]])
            self.processor.thresh_zcr = (zcr_vowels + zcr_frics) / 2

            # 4. Centroid Vowel Split (OOO vs EEE)
            # OOO should be lower freq, EEE higher freq
            cent_o = np.mean([x[2] for x in results["OOO"]])
            cent_e = np.mean([x[2] for x in results["EEE"]])
            self.processor.thresh_vowel_cent = (cent_o + cent_e) / 2

            # 5. Centroid Fricative Split (SHHH vs SSSS)
            # SHHH should be lower (more "hush"), SSSS higher (more "hiss")
            cent_sh = np.mean([x[2] for x in results["SHHH"]])
            cent_s = np.mean([x[2] for x in results["SSSS"]])
            self.processor.thresh_fric_cent = (cent_sh + cent_s) / 2

            print(f"CALIBRATION RESULTS:\nSilence: {self.processor.thresh_silence}\nZCR Split: {self.processor.thresh_zcr}\nVowel Split: {self.processor.thresh_vowel_cent}\nFric Split: {self.processor.thresh_fric_cent}")
            
            self.processor.calibrating = False; self.processor.stop()
            top.destroy()
            messagebox.showinfo("Done", "Calibration Complete!\nSystem Adapted to your Voice.")

        self.after(100, lambda: run_step(0))

    def on_close(self):
        self.processor.stop()
        self.destroy()

if __name__ == "__main__":
    app = VoiceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()