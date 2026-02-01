import sys
import time
import numpy as np
import pyaudio
import pydirectinput
from scipy import signal
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import pyqtgraph as pg

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

class AudioWorker(QThread):
    # Signals to send data back to the GUI
    update_plots = pyqtSignal(np.ndarray, np.ndarray, np.ndarray) # Raw, Filtered, FFT
    update_status = pyqtSignal(str, float, float) # Status text, Vol, Pitch
    
    def __init__(self):
        super().__init__()
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = True
        self.paused = False
        self.recalibrating = False
        self.pressed = {'up':False, 'down':False, 'left':False, 'right':False, 'enter':False}
        
        # DSP Filter Design (High Pass > 100Hz)
        # This removes DC offset and low rumble noise
        sos = signal.butter(10, 100, 'hp', fs=RATE, output='sos')
        self.filter_sos = sos

        # Initial Thresholds
        self.silence_thresh = 500
        self.respawn_thresh = 15000
        self.pitch_thresh = 0 
        self.ratio_oe = 1.5 
        self.ratio_ssh = 3.0

    def select_device(self):
        # Auto-select default device for smoother UI startup
        # In a real app, you might want a dropdown.
        return self.p.get_default_input_device_info()['index']

    def run(self):
        dev_index = self.select_device()
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                  input_device_index=dev_index, frames_per_buffer=CHUNK)
        
        # Initial Calibration
        self.run_calibration()

        while self.running:
            if self.paused or self.recalibrating:
                time.sleep(0.1)
                continue

            try:
                # 1. Capture Raw Audio
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                raw_audio = np.frombuffer(data, dtype=np.int16).astype(np.float64) * GAIN
                
                # 2. Apply Digital Filter (High Pass) - THE NEW ADDITION
                # This makes the "Filtered" graph look different from "Raw"
                filtered_audio = signal.sosfilt(self.filter_sos, raw_audio)

                # 3. FFT Analysis (Frequency Domain)
                window = np.hamming(len(filtered_audio))
                fft_complex = np.fft.rfft(filtered_audio * window)
                fft_mag = np.abs(fft_complex)
                
                # Process Control Logic
                self.process_logic(filtered_audio, fft_mag)
                
                # Send data to GUI
                self.update_plots.emit(raw_audio, filtered_audio, fft_mag)

            except Exception as e:
                print(f"Error: {e}")

        self.stop_stream()

    def process_logic(self, audio, mag):
        # Calculate Volume (RMS)
        vol = np.sqrt(np.mean(audio**2))
        
        # Frequency Bands
        i_pitch = (int(100/(RATE/CHUNK)), int(300/(RATE/CHUNK)))
        e_pitch = np.sum(mag[i_pitch[0]:i_pitch[1]])

        i_low = (int(300/(RATE/CHUNK)), int(800/(RATE/CHUNK)))
        e_low = np.sum(mag[i_low[0]:i_low[1]])

        i_mid = (int(2000/(RATE/CHUNK)), int(4000/(RATE/CHUNK)))
        e_mid = np.sum(mag[i_mid[0]:i_mid[1]])

        i_high = (int(5000/(RATE/CHUNK)), int(10000/(RATE/CHUNK)))
        e_high = np.sum(mag[i_high[0]:i_high[1]])

        up, down, left, right = False, False, False, False
        status_msg = "IDLE..."

        # Logic Tree
        if vol > self.respawn_thresh:
            pydirectinput.press(KEY_RESPAWN)
            status_msg = ">>> CLAP / RESPAWN <<<"
        elif vol > self.silence_thresh:
            has_pitch = e_pitch > self.pitch_thresh

            if has_pitch:
                # Vowels
                ratio = e_mid / (e_low + 1)
                if ratio > self.ratio_oe:
                    left = True; up = True
                    status_msg = f"LEFT (EEE) [Ratio: {ratio:.1f}]"
                else:
                    right = True; up = True
                    status_msg = f"RIGHT (OOO) [Ratio: {ratio:.1f}]"
            else:
                # Noise
                ratio = e_high / (e_mid + 1)
                if ratio > self.ratio_ssh or ratio > 5.0:
                    up = True
                    status_msg = f"GAS (SSS) [Ratio: {ratio:.1f}]"
                else:
                    down = True
                    status_msg = f"BRAKE (SHH) [Ratio: {ratio:.1f}]"
        
        self.apply_keys(up, down, left, right)
        self.update_status.emit(status_msg, vol, e_pitch)

    def apply_keys(self, up, down, left, right):
        keys = [(KEY_ACCEL, up), (KEY_BRAKE, down), (KEY_LEFT, left), (KEY_RIGHT, right)]
        for k, s in keys:
            if s and not self.pressed[k]:
                pydirectinput.keyDown(k); self.pressed[k]=True
            elif not s and self.pressed[k]:
                pydirectinput.keyUp(k); self.pressed[k]=False

    def run_calibration(self):
        # A simplified, non-blocking calibration could go here
        # For now, we use defaults or pre-calibrated values to avoid GUI freeze
        # In a full app, you'd trigger a specific state.
        pass

    def stop_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSP Voice Controller - Realtime Analysis")
        self.resize(500, 900) # Vertical layout as requested

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # 1. Raw Audio Plot
        self.lbl_raw = QLabel("Raw Audio (Microphone Input)")
        self.lbl_raw.setStyleSheet("font-weight: bold; color: #555;")
        layout.addWidget(self.lbl_raw)
        
        self.plot_raw = pg.PlotWidget()
        self.plot_raw.setYRange(-20000, 20000)
        self.plot_raw.hideAxis('bottom')
        self.curve_raw = self.plot_raw.plot(pen=pg.mkPen('#00BFFF', width=1))
        layout.addWidget(self.plot_raw)

        # 2. Filtered Audio Plot
        self.lbl_filt = QLabel("Filtered Audio (High-Pass > 100Hz)")
        self.lbl_filt.setStyleSheet("font-weight: bold; color: #555;")
        layout.addWidget(self.lbl_filt)

        self.plot_filt = pg.PlotWidget()
        self.plot_filt.setYRange(-20000, 20000)
        self.plot_filt.hideAxis('bottom')
        self.curve_filt = self.plot_filt.plot(pen=pg.mkPen('#00FA9A', width=1))
        layout.addWidget(self.plot_filt)

        # 3. FFT Spectrum Plot (Frequency Domain)
        self.lbl_fft = QLabel("FFT Spectrum (Frequency Domain)")
        self.lbl_fft.setStyleSheet("font-weight: bold; color: #555;")
        layout.addWidget(self.lbl_fft)

        self.plot_fft = pg.PlotWidget()
        self.plot_fft.setLabel('bottom', 'Frequency Bins')
        self.plot_fft.setRange(xRange=[0, 300], yRange=[0, 1000000]) # Zoom in on useful freqs
        self.curve_fft = self.plot_fft.plot(pen=pg.mkPen('#FF69B4', width=2), fillLevel=0, brush=(255,105,180,50))
        layout.addWidget(self.plot_fft)

        # 4. Status / Control Display
        self.lbl_status = QLabel("IDLE")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("""
            background-color: #333; 
            color: #FFF; 
            font-size: 24px; 
            font-weight: bold; 
            border-radius: 10px; 
            padding: 15px;
        """)
        layout.addWidget(self.lbl_status)

        # 5. Instructions Label
        self.lbl_help = QLabel("[ESC] Quit  |  [SPACE] Pause/Resume  |  [R] Recalibrate")
        self.lbl_help.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_help.setStyleSheet("color: #777; font-size: 12px;")
        layout.addWidget(self.lbl_help)

        # Start Worker Thread
        self.worker = AudioWorker()
        self.worker.update_plots.connect(self.update_graphs)
        self.worker.update_status.connect(self.update_labels)
        self.worker.start()

    def update_graphs(self, raw, filt, fft):
        self.curve_raw.setData(raw)
        self.curve_filt.setData(filt)
        self.curve_fft.setData(fft[:300]) # Plotting first 300 bins is usually enough for voice

    def update_labels(self, text, vol, pitch):
        self.lbl_status.setText(text)
        # Dynamic color changing based on detection
        if "LEFT" in text: self.lbl_status.setStyleSheet("background-color: #2E8B57; color: white; font-size: 24px; padding: 15px; border-radius:10px;")
        elif "RIGHT" in text: self.lbl_status.setStyleSheet("background-color: #2E8B57; color: white; font-size: 24px; padding: 15px; border-radius:10px;")
        elif "GAS" in text: self.lbl_status.setStyleSheet("background-color: #DAA520; color: white; font-size: 24px; padding: 15px; border-radius:10px;")
        elif "BRAKE" in text: self.lbl_status.setStyleSheet("background-color: #B22222; color: white; font-size: 24px; padding: 15px; border-radius:10px;")
        else: self.lbl_status.setStyleSheet("background-color: #333; color: white; font-size: 24px; padding: 15px; border-radius:10px;")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.worker.running = False
            self.worker.wait()
            self.close()
        elif event.key() == Qt.Key.Key_Space:
            self.worker.paused = not self.worker.paused
            state = "PAUSED" if self.worker.paused else "RESUMED"
            self.lbl_status.setText(state)
        elif event.key() == Qt.Key.Key_R:
            self.lbl_status.setText("RECALIBRATING... (Not Implemented)")
            # You can trigger the calibration function here in the worker thread

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())