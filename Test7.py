import sys
import time
import numpy as np
import pyaudio
import pydirectinput
from scipy import signal
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QLabel, QProgressBar, QStackedWidget, QComboBox, QPushButton, QHBoxLayout)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
import pyqtgraph as pg

# --- CONFIGURATION ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
GAIN = 5.0

# CONTROLS
KEY_ACCEL = 'w' # up
KEY_BRAKE = 's' # down
KEY_LEFT = 'a' # left
KEY_RIGHT = 'd' # right
KEY_RESPAWN = 'q' # enter

class AudioWorker(QThread):
    # Signals
    update_plots = pyqtSignal(np.ndarray, np.ndarray, np.ndarray) # Raw, Filtered, FFT
    update_status = pyqtSignal(str, float, float) # Status, Vol, Pitch
    calibration_progress = pyqtSignal(int, str) # Progress %, Message
    calibration_finished = pyqtSignal(dict) # Returns calculated thresholds
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = True
        self.paused = False
        self.device_index = None
        
        # Mode: 'IDLE', 'CALIBRATING', 'GAME'
        self.mode = 'IDLE' 
        self.calib_step = 0
        self.calib_buffer = []
        self.calib_target_samples = 50 # How many chunks to measure per step
        
        #self.pressed = {'up':False, 'down':False, 'left':False, 'right':False, 'enter':False}
        # This automatically adds whatever keys you set in the CONTROLS section
        self.pressed = {
            KEY_ACCEL: False, 
            KEY_BRAKE: False, 
            KEY_LEFT: False, 
            KEY_RIGHT: False, 
            KEY_RESPAWN: False
        }
        
        # DSP Filter Design (High Pass > 100Hz)
        self.filter_sos = signal.butter(10, 100, 'hp', fs=RATE, output='sos')

        # Thresholds (Will be overwritten by calibration)
        self.thresh = {
            'silence': 500,
            'respawn': 15000,
            'pitch': 0,
            'ratio_oe': 1.5,
            'ratio_ssh': 3.0
        }

    def set_device(self, index):
        self.device_index = index

    def start_calibration_step(self, step_num):
        """Prepares the worker to collect data for a specific calibration step"""
        self.calib_step = step_num
        self.calib_buffer = [] # Clear buffer
        self.mode = 'CALIBRATING'

    def run(self):
        if self.device_index is None:
            self.error_occurred.emit("No Device Selected")
            return

        try:
            self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                      input_device_index=self.device_index, frames_per_buffer=CHUNK)
        except Exception as e:
            self.error_occurred.emit(str(e))
            return

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            try:
                # 1. Capture & Process
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                raw_audio = np.frombuffer(data, dtype=np.int16).astype(np.float64) * GAIN
                filtered_audio = signal.sosfilt(self.filter_sos, raw_audio)
                
                # FFT
                window = np.hamming(len(raw_audio))
                fft_complex = np.fft.rfft(raw_audio * window)
                fft_mag = np.abs(fft_complex)
                
                # Calculates metrics for the current frame
                metrics = self.calculate_metrics(raw_audio, fft_mag)
                
                # 2. Handle Modes
                if self.mode == 'CALIBRATING':
                    self.handle_calibration(metrics)
                    status_msg = f"CALIBRATING... {len(self.calib_buffer)}/{self.calib_target_samples}"
                
                elif self.mode == 'GAME':
                    status_msg = self.handle_game_logic(metrics)
                
                else:
                    status_msg = "IDLE"

                # 3. Update GUI
                self.update_plots.emit(raw_audio, filtered_audio, fft_mag)
                self.update_status.emit(status_msg, metrics['vol'], metrics['pitch'])

            except Exception as e:
                print(f"Stream Error: {e}")
                break

        self.stop_stream()

    def calculate_metrics(self, audio, mag):
        vol = np.sqrt(np.mean(audio**2))
        
        # Frequency Bands
        def get_band_energy(low_freq, high_freq):
            idx_lo = int(low_freq / (RATE/CHUNK))
            idx_hi = int(high_freq / (RATE/CHUNK))
            return np.sum(mag[idx_lo:idx_hi])

        e_pitch = get_band_energy(100, 300)
        e_low = get_band_energy(300, 800)
        e_mid = get_band_energy(2000, 4000)
        e_high = get_band_energy(5000, 10000)
        
        return {
            'vol': vol, 'pitch': e_pitch, 
            'low': e_low, 'mid': e_mid, 'high': e_high
        }

    def handle_calibration(self, m):
        # Collects N samples then processes them
        if len(self.calib_buffer) < self.calib_target_samples:
            self.calib_buffer.append(m)
            progress = int((len(self.calib_buffer) / self.calib_target_samples) * 100)
            self.calibration_progress.emit(progress, "")
        else:
            # Step Complete - Calculate Stats
            self.mode = 'IDLE' # Pause collection
            self.calibration_finished.emit(self.process_calibration_stats())

    def process_calibration_stats(self):
        # Average the collected buffer
        avg_vol = np.mean([x['vol'] for x in self.calib_buffer])
        
        # --- NEW: Also calculate the PEAK volume (for Clap) ---
        max_vol = np.max([x['vol'] for x in self.calib_buffer]) 
        
        avg_pitch = np.mean([x['pitch'] for x in self.calib_buffer])
        
        # Ratios
        r_oe_list = [(x['mid']/(x['low']+1)) for x in self.calib_buffer]
        r_ssh_list = [(x['high']/(x['mid']+1)) for x in self.calib_buffer]
        
        return {
            'step': self.calib_step,
            'vol': avg_vol,
            'max_vol': max_vol,  # <--- Passing the Peak Volume
            'pitch': avg_pitch,
            'r_oe': np.median(r_oe_list) if r_oe_list else 0,
            'r_ssh': np.median(r_ssh_list) if r_ssh_list else 0
        }

    def handle_game_logic(self, m):
        up, down, left, right = False, False, False, False
        status = "..."

        if m['vol'] > self.thresh['respawn']:
            pydirectinput.press(KEY_RESPAWN)
            return ">>> CLAP / RESPAWN <<<"

        if m['vol'] > self.thresh['silence']:
            has_pitch = m['pitch'] > self.thresh['pitch']

            if has_pitch:
                # Vowels (O vs E)
                ratio = m['mid'] / (m['low'] + 1)
                if ratio > self.thresh['ratio_oe']:
                    left = True; up = True
                    status = f"LEFT (EEE)"
                else:
                    right = True; up = True
                    status = f"RIGHT (OOO)"
            else:
                # Noise (S vs SH)
                ratio = m['high'] / (m['mid'] + 1)
                if ratio > self.thresh['ratio_ssh'] or ratio > 5.0:
                    up = True
                    status = f"GAS (SSS)"
                else:
                    down = True
                    status = f"BRAKE (SHH)"
        
        self.apply_keys(up, down, left, right)
        return status

    def apply_keys(self, up, down, left, right):
        keys = [(KEY_ACCEL, up), (KEY_BRAKE, down), (KEY_LEFT, left), (KEY_RIGHT, right)]
        for k, s in keys:
            if s and not self.pressed[k]:
                pydirectinput.keyDown(k); self.pressed[k]=True
            elif not s and self.pressed[k]:
                pydirectinput.keyUp(k); self.pressed[k]=False
    
    def release_all_keys(self):
        for k in self.pressed:
            if self.pressed[k]:
                pydirectinput.keyUp(k)
                self.pressed[k] = False

    def stop_stream(self):
        self.release_all_keys()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSP Voice Controller - Ultimate UI")
        self.resize(500, 950)

        # Worker Thread
        self.worker = AudioWorker()
        self.worker.update_plots.connect(self.update_graphs)
        self.worker.update_status.connect(self.update_status_label)
        self.worker.calibration_progress.connect(self.update_calib_progress)
        self.worker.calibration_finished.connect(self.on_calib_step_complete)

        # --- CENTRAL WIDGET ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        # --- 1. GRAPHS (Always Visible) ---
        self.setup_graphs()

        # --- 2. CONTROL STACK (Swaps between Setup, Calibration, Game) ---
        self.stack = QStackedWidget()
        self.main_layout.addWidget(self.stack)

        # Setup Screens
        self.setup_device_screen()
        self.setup_calibration_screen()
        self.setup_game_screen()
        
        # Start at device selection
        self.refresh_devices()
        self.stack.setCurrentIndex(0)

    def setup_graphs(self):
        # Raw
        self.main_layout.addWidget(QLabel("Raw Input (Time Domain)"))
        self.plot_raw = pg.PlotWidget()
        self.plot_raw.setYRange(-20000, 20000); self.plot_raw.hideAxis('bottom'); self.plot_raw.setFixedHeight(150)
        self.curve_raw = self.plot_raw.plot(pen=pg.mkPen('#00BFFF', width=1))
        self.main_layout.addWidget(self.plot_raw)

        # Filtered
        self.main_layout.addWidget(QLabel("Filtered Input (High-Pass > 100Hz)"))
        self.plot_filt = pg.PlotWidget()
        self.plot_filt.setYRange(-20000, 20000); self.plot_filt.hideAxis('bottom'); self.plot_filt.setFixedHeight(150)
        self.curve_filt = self.plot_filt.plot(pen=pg.mkPen('#00FA9A', width=1))
        self.main_layout.addWidget(self.plot_filt)

        # FFT
        self.main_layout.addWidget(QLabel("Frequency Spectrum (FFT)"))
        self.plot_fft = pg.PlotWidget()
        self.plot_fft.setRange(xRange=[0, 300], yRange=[0, 1000000]); self.plot_fft.setFixedHeight(150)
        self.curve_fft = self.plot_fft.plot(pen=pg.mkPen('#FF69B4', width=2), fillLevel=0, brush=(255,105,180,50))
        self.main_layout.addWidget(self.plot_fft)

    def setup_device_screen(self):
        self.page_device = QWidget()
        layout = QVBoxLayout(self.page_device)
        
        lbl = QLabel("STEP 1: SELECT MICROPHONE")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        layout.addWidget(lbl)

        self.combo_devices = QComboBox()
        self.combo_devices.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(self.combo_devices)

        btn = QPushButton("CONFIRM & START CALIBRATION")
        btn.setStyleSheet("background-color: #2E8B57; color: white; font-size: 14px; padding: 15px; font-weight: bold;")
        btn.clicked.connect(self.start_calibration_sequence)
        layout.addWidget(btn)
        
        self.stack.addWidget(self.page_device)

    def setup_calibration_screen(self):
        self.page_calib = QWidget()
        layout = QVBoxLayout(self.page_calib)

        self.lbl_calib_instr = QLabel("GET READY...")
        self.lbl_calib_instr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_calib_instr.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFA500;")
        layout.addWidget(self.lbl_calib_instr)

        self.progress_calib = QProgressBar()
        self.progress_calib.setStyleSheet("height: 30px;")
        layout.addWidget(self.progress_calib)

        self.btn_calib_action = QPushButton("START MEASUREMENT")
        self.btn_calib_action.setStyleSheet("background-color: #4682B4; color: white; font-size: 16px; padding: 15px;")
        self.btn_calib_action.clicked.connect(self.trigger_calib_step)
        layout.addWidget(self.btn_calib_action)

        self.stack.addWidget(self.page_calib)

    def setup_game_screen(self):
        self.page_game = QWidget()
        layout = QVBoxLayout(self.page_game)

        self.lbl_status = QLabel("IDLE")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("background-color: #333; color: white; font-size: 28px; font-weight: bold; border-radius: 10px; padding: 20px;")
        layout.addWidget(self.lbl_status)

        help_txt = QLabel("[SPACE] Pause  |  [ESC] Quit  |  [R] Recalibrate (Reset)")
        help_txt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(help_txt)

        self.stack.addWidget(self.page_game)

    def refresh_devices(self):
        p = pyaudio.PyAudio()
        self.combo_devices.clear()
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                self.combo_devices.addItem(f"{info['index']}: {info['name']}", info['index'])
        p.terminate()

    # --- LOGIC FLOW ---

    def start_calibration_sequence(self):
        # User picked device, start thread
        idx = self.combo_devices.currentData()
        self.worker.set_device(idx)
        if not self.worker.isRunning():
            self.worker.start()
        
        self.stack.setCurrentIndex(1) # Show Calib Screen
        self.calib_stage = 0
        self.next_calib_stage()

    def next_calib_stage(self):
        stages = [
            ("1. SILENCE", "Stay completely silent..."),
            ("2. 'OOO' (Turn Right)", "Make a deep 'OOO' sound..."),
            ("3. 'EEE' (Turn Left)", "Make a bright 'EEE' sound..."),
            ("4. 'SHH' (Brake)", "Make a 'SHHH' noise..."),
            ("5. 'SSS' (Gas)", "Make a sharp 'SSSS' noise..."),
            ("6. CLAP (Respawn)", "Clap your hands loudly...")
        ]
        
        if self.calib_stage < len(stages):
            title, desc = stages[self.calib_stage]
            self.lbl_calib_instr.setText(f"{title}\n{desc}")
            self.progress_calib.setValue(0)
            self.btn_calib_action.setText("CLICK TO CAPTURE")
            self.btn_calib_action.setEnabled(True)
        else:
            self.finish_calibration()

    def trigger_calib_step(self):
        self.btn_calib_action.setEnabled(False)
        self.btn_calib_action.setText("MEASURING...")
        self.worker.start_calibration_step(self.calib_stage)

    def update_calib_progress(self, val, msg):
        self.progress_calib.setValue(val)

    def on_calib_step_complete(self, data):
        # Save data based on stage
        stage = self.calib_stage
        
        if stage == 0: # Silence
            self.worker.thresh['silence'] = max(data['vol'] * 2.0, 500)
            print(f"Silence Floor: {self.worker.thresh['silence']}")
        
        elif stage == 1: # OOO
            self.calib_o_pitch = data['pitch']
            self.calib_o_ratio = data['r_oe']
        
        elif stage == 2: # EEE
            # Set Pitch Thresh
            avg_pitch = min(self.calib_o_pitch, data['pitch'])
            self.worker.thresh['pitch'] = avg_pitch * 0.4
            
            # Set O/E Boundary
            self.worker.thresh['ratio_oe'] = (self.calib_o_ratio + data['r_oe']) / 2
            print(f"Pitch Gate: {self.worker.thresh['pitch']} | OE Split: {self.worker.thresh['ratio_oe']}")

        elif stage == 3: # SHH
            self.calib_sh_ratio = data['r_ssh']

        elif stage == 4: # SSS
            self.worker.thresh['ratio_ssh'] = (self.calib_sh_ratio + data['r_ssh']) / 2
            print(f"S/SH Split: {self.worker.thresh['ratio_ssh']}")

        elif stage == 5: # CLAP
            # --- FIX IS HERE ---
            # We use 'max_vol' (Peak) instead of 'vol' (Average)
            # This ensures the threshold is set to the loudness of the clap itself,
            # not the silence that came after it.
            self.worker.thresh['respawn'] = data['max_vol'] * 0.8
            print(f"Clap Thresh: {self.worker.thresh['respawn']}")

        self.calib_stage += 1
        self.next_calib_stage()
        
    def finish_calibration(self):
        self.worker.mode = 'GAME'
        self.stack.setCurrentIndex(2) # Go to Game Screen

    def reset_program(self):
        self.worker.mode = 'IDLE'
        self.worker.release_all_keys()
        self.stack.setCurrentIndex(0) # Back to device select
        self.refresh_devices()

    # --- UI UPDATES ---

    def update_graphs(self, raw, filt, fft):
        self.curve_raw.setData(raw)
        self.curve_filt.setData(filt)
        self.curve_fft.setData(fft[:300])

    def update_status_label(self, text, vol, pitch):
        if self.worker.mode == 'GAME':
            self.lbl_status.setText(text)
            # Simple color coding
            color = "#333"
            if "LEFT" in text or "RIGHT" in text: color = "#2E8B57" # Green
            elif "GAS" in text: color = "#DAA520" # Gold
            elif "BRAKE" in text: color = "#B22222" # Red
            self.lbl_status.setStyleSheet(f"background-color: {color}; color: white; font-size: 28px; font-weight: bold; border-radius: 10px; padding: 20px;")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.worker.running = False
            self.close()
        elif event.key() == Qt.Key.Key_Space:
            self.worker.paused = not self.worker.paused
            self.lbl_status.setText("PAUSED" if self.worker.paused else "RESUMED")
        elif event.key() == Qt.Key.Key_R:
            self.reset_program()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())