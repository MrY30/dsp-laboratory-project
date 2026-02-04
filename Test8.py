import sys
import time
import numpy as np
import pyaudio
import pydirectinput
from scipy import signal
from collections import deque

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QPushButton, QComboBox, QProgressBar, QStackedWidget
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import pyqtgraph as pg

# ==========================
# DSP CONFIGURATION
# ==========================
CHUNK = 1024
RATE = 44100
GAIN = 5.0

# Feature smoothing window
SMOOTH_WIN = 7
DECISION_HOLD = 0.15  # seconds

# ==========================
# GAME CONTROLS
# ==========================
KEY_ACCEL = 'w'
KEY_BRAKE = 's'
KEY_LEFT = 'a'
KEY_RIGHT = 'd'
KEY_RESPAWN = 'q'

# ==========================
# AUDIO WORKER
# ==========================
class AudioWorker(QThread):
    update_gui = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.device = None

        self.p = pyaudio.PyAudio()
        self.stream = None

        # DSP filters
        self.hp = signal.butter(8, 100, 'hp', fs=RATE, output='sos')

        # Feature buffers
        self.hist = {
            'vol': deque(maxlen=SMOOTH_WIN),
            'pitch': deque(maxlen=SMOOTH_WIN),
            'low': deque(maxlen=SMOOTH_WIN),
            'mid': deque(maxlen=SMOOTH_WIN),
            'high': deque(maxlen=SMOOTH_WIN)
        }

        # Thresholds
        self.thresh = {
            'silence': 600,
            'pitch': 0,
            'ratio_oe': 1.5,
            'ratio_ssh': 3.0,
            'clap': 15000
        }

        self.last_action = None
        self.last_time = 0

        self.pressed = {
            KEY_ACCEL: False,
            KEY_BRAKE: False,
            KEY_LEFT: False,
            KEY_RIGHT: False
        }

    def set_device(self, idx):
        self.device = idx

    def run(self):
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                input_device_index=self.device,
                frames_per_buffer=CHUNK
            )
        except Exception as e:
            self.error.emit(str(e))
            return

        while self.running:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float64) * GAIN
            audio = signal.sosfilt(self.hp, audio)

            vol = np.sqrt(np.mean(audio ** 2))
            fft = np.abs(np.fft.rfft(audio * np.hamming(len(audio))))

            def band(lo, hi):
                return np.sum(fft[int(lo / (RATE / CHUNK)):int(hi / (RATE / CHUNK))])

            feats = {
                'vol': vol,
                'pitch': band(100, 300),
                'low': band(300, 800),
                'mid': band(2000, 4000),
                'high': band(5000, 10000)
            }

            for k in feats:
                self.hist[k].append(feats[k])

            sm = {k: np.median(self.hist[k]) for k in self.hist}

            action = self.decide(sm)
            self.apply(action)

            self.update_gui.emit({
                'action': action,
                'vol': sm['vol'],
                'pitch': sm['pitch']
            })

        self.cleanup()

    # ==========================
    # DECISION LOGIC (STABLE)
    # ==========================
    def decide(self, m):
        now = time.time()
        if now - self.last_time < DECISION_HOLD:
            return self.last_action

        if m['vol'] > self.thresh['clap']:
            self.last_action = 'RESPAWN'
            self.last_time = now
            return 'RESPAWN'

        if m['vol'] < self.thresh['silence']:
            self.last_action = 'IDLE'
            return 'IDLE'

        if m['pitch'] > self.thresh['pitch']:
            ratio = m['mid'] / (m['low'] + 1)
            act = 'LEFT' if ratio > self.thresh['ratio_oe'] else 'RIGHT'
        else:
            ratio = m['high'] / (m['mid'] + 1)
            act = 'GAS' if ratio > self.thresh['ratio_ssh'] else 'BRAKE'

        self.last_action = act
        self.last_time = now
        return act

    # ==========================
    # KEY HANDLING
    # ==========================
    def apply(self, act):
        mapping = {
            'LEFT': (True, False, True, False),
            'RIGHT': (True, False, False, True),
            'GAS': (True, False, False, False),
            'BRAKE': (False, True, False, False),
            'IDLE': (False, False, False, False)
        }

        if act == 'RESPAWN':
            pydirectinput.press(KEY_RESPAWN)
            return

        up, down, left, right = mapping.get(act, (False, False, False, False))
        keys = [(KEY_ACCEL, up), (KEY_BRAKE, down),
                (KEY_LEFT, left), (KEY_RIGHT, right)]

        for k, s in keys:
            if s and not self.pressed[k]:
                pydirectinput.keyDown(k)
                self.pressed[k] = True
            elif not s and self.pressed[k]:
                pydirectinput.keyUp(k)
                self.pressed[k] = False

    def cleanup(self):
        for k in self.pressed:
            if self.pressed[k]:
                pydirectinput.keyUp(k)
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

# ==========================
# GUI
# ==========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSP Voice Controller – Improved")
        self.resize(500, 400)

        self.worker = AudioWorker()
        self.worker.update_gui.connect(self.update_status)

        layout = QVBoxLayout()
        self.lbl = QLabel("IDLE")
        self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl.setStyleSheet("font-size: 28px; padding: 20px;")
        layout.addWidget(self.lbl)

        btn = QPushButton("START")
        btn.clicked.connect(self.start)
        layout.addWidget(btn)

        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

    def start(self):
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            if p.get_device_info_by_index(i)['maxInputChannels'] > 0:
                self.worker.set_device(i)
                break
        p.terminate()
        self.worker.start()

    def update_status(self, d):
        self.lbl.setText(d['action'])

    def closeEvent(self, e):
        self.worker.running = False
        e.accept()

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
