import sys
import time
import numpy as np
import pyaudio
import pydirectinput
from scipy import signal
from collections import deque

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QPushButton, QComboBox, QProgressBar, QStackedWidget
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
import pyqtgraph as pg

# =====================
# CONFIGURATION
# =====================
CHUNK = 1024
RATE = 44100
GAIN = 5.0

SMOOTH_WIN = 7
DECISION_HOLD = 0.12
UI_REFRESH_MS = 33

KEY_ACCEL = 'w'
KEY_BRAKE = 's'
KEY_LEFT = 'a'
KEY_RIGHT = 'd'
KEY_RESPAWN = 'q'

# =====================
# AUDIO THREAD
# =====================
class AudioWorker(QThread):
    update_data = pyqtSignal(dict)
    calib_progress = pyqtSignal(int)
    calib_done = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = True
        self.mode = 'IDLE'
        self.device = None

        self.p = pyaudio.PyAudio()
        self.stream = None

        self.hp = signal.butter(8, 100, 'hp', fs=RATE, output='sos')

        self.hist = {k: deque(maxlen=SMOOTH_WIN)
                     for k in ['vol','pitch','low','mid','high']}

        self.calib_buf = []
        self.calib_target = 60
        self.calib_step = 0

        self.thresh = {
            'silence': 500,
            'pitch': 0,
            'ratio_oe': 1.5,
            'ratio_ssh': 3.0,
            'clap': 15000
        }

        self.last_action = 'IDLE'
        self.last_time = 0

        self.pressed = {KEY_ACCEL:False, KEY_BRAKE:False,
                        KEY_LEFT:False, KEY_RIGHT:False}

    def set_device(self, idx):
        self.device = idx

    def start_calibration(self, step):
        self.calib_step = step
        self.calib_buf.clear()
        self.mode = 'CALIB'

    def run(self):
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            input_device_index=self.device,
            frames_per_buffer=CHUNK
        )

        while self.running:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float64) * GAIN
            audio = signal.sosfilt(self.hp, audio)

            vol = np.sqrt(np.mean(audio**2))
            fft = np.abs(np.fft.rfft(audio * np.hamming(len(audio))))

            def band(lo, hi):
                return np.sum(fft[int(lo/(RATE/CHUNK)):int(hi/(RATE/CHUNK))])

            m = {
                'vol': vol,
                'pitch': band(100,300),
                'low': band(300,800),
                'mid': band(2000,4000),
                'high': band(5000,10000),
                'raw': audio,
                'fft': fft
            }

            if self.mode == 'CALIB':
                self.calib_buf.append(m)
                self.calib_progress.emit(
                    int(100 * len(self.calib_buf) / self.calib_target)
                )
                if len(self.calib_buf) >= self.calib_target:
                    self.finish_calibration()
                continue

            for k in self.hist:
                self.hist[k].append(m[k])

            sm = {k: np.median(self.hist[k]) for k in self.hist}
            action = self.decide(sm)
            self.apply_keys(action)

            m['action'] = action
            self.update_data.emit(m)

        self.cleanup()

    def finish_calibration(self):
        vols = [x['vol'] for x in self.calib_buf]
        pitch = np.median([x['pitch'] for x in self.calib_buf])
        r_oe = np.median([x['mid']/(x['low']+1) for x in self.calib_buf])
        r_ssh = np.median([x['high']/(x['mid']+1) for x in self.calib_buf])

        self.mode = 'IDLE'
        self.calib_done.emit({
            'step': self.calib_step,
            'vol': np.mean(vols),
            'max_vol': np.max(vols),
            'pitch': pitch,
            'r_oe': r_oe,
            'r_ssh': r_ssh
        })

    def decide(self, m):
        now = time.time()
        if now - self.last_time < DECISION_HOLD:
            return self.last_action

        if m['vol'] > self.thresh['clap']:
            self.last_action = 'RESPAWN'
            self.last_time = now
            return 'RESPAWN'

        if m['vol'] < self.thresh['silence']:
            return 'IDLE'

        if m['pitch'] > self.thresh['pitch']:
            act = 'LEFT' if m['mid']/(m['low']+1) > self.thresh['ratio_oe'] else 'RIGHT'
        else:
            act = 'GAS' if m['high']/(m['mid']+1) > self.thresh['ratio_ssh'] else 'BRAKE'

        self.last_action = act
        self.last_time = now
        return act

    def apply_keys(self, act):
        mapping = {
            'LEFT': (True,False,True,False),
            'RIGHT': (True,False,False,True),
            'GAS': (True,False,False,False),
            'BRAKE': (False,True,False,False),
            'IDLE': (False,False,False,False)
        }

        if act == 'RESPAWN':
            pydirectinput.press(KEY_RESPAWN)
            return

        up, down, left, right = mapping.get(act,(0,0,0,0))
        for k, s in zip([KEY_ACCEL,KEY_BRAKE,KEY_LEFT,KEY_RIGHT],
                        [up,down,left,right]):
            if s and not self.pressed[k]:
                pydirectinput.keyDown(k)
                self.pressed[k]=True
            elif not s and self.pressed[k]:
                pydirectinput.keyUp(k)
                self.pressed[k]=False

    def cleanup(self):
        for k in self.pressed:
            if self.pressed[k]:
                pydirectinput.keyUp(k)
        self.stream.close()
        self.p.terminate()
