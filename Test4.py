import pyaudio
import numpy as np
import pydirectinput
import time
import sys
from collections import deque

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

class VoiceController:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = True
        self.pressed = {'up':False, 'down':False, 'left':False, 'right':False, 'enter':False}

        # Thresholds
        self.silence_thresh = 500
        self.respawn_thresh = 10000
        
        # CRITICAL: This separates EEE from SHHH
        self.pitch_thresh = 0 
        
        # Decision Boundaries
        self.ratio_oe = 1.5 
        self.ratio_ssh = 3.0

    def select_device(self):
        print("\n--- MICROPHONE SELECTION ---")
        info = self.p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = self.p.get_device_info_by_host_api_device_index(0, i).get('name')
                print(f"ID {i} - {name}")
        print("----------------------------")
        while True:
            try: return int(input("Enter ID: "))
            except: pass

    def start(self):
        dev_index = self.select_device()
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                  input_device_index=dev_index, frames_per_buffer=CHUNK)
        self.run_calibration()
        print("\n" + "="*40 + "\n     CONTROLLER ACTIVE\n     (Ctrl+C to Stop)\n" + "="*40)
        try:
            while True: self.process_audio()
        except KeyboardInterrupt:
            self.stop()

    def get_spectrum(self):
        try:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float64) * GAIN
            
            # Volume
            rms = np.sqrt(np.mean(audio**2))
            
            # FFT
            window = np.hamming(len(audio))
            fft = np.fft.rfft(audio * window)
            mag = np.abs(fft)
            
            # --- THE 4 BANDS ---
            # 1. PITCH (100-300Hz): The "Hum" of vocal cords.
            # SHHH and SSSS have almost ZERO energy here.
            i_pitch = (int(100/(RATE/CHUNK)), int(300/(RATE/CHUNK)))
            e_pitch = np.sum(mag[i_pitch[0]:i_pitch[1]])

            # 2. LOW (300-800Hz): The body of 'O'
            i_low = (int(300/(RATE/CHUNK)), int(800/(RATE/CHUNK)))
            e_low = np.sum(mag[i_low[0]:i_low[1]])

            # 3. MID (2000-4000Hz): The body of 'E' and 'SH'
            i_mid = (int(2000/(RATE/CHUNK)), int(4000/(RATE/CHUNK)))
            e_mid = np.sum(mag[i_mid[0]:i_mid[1]])

            # 4. HIGH (5000-10000Hz): The sharpness of 'S'
            i_high = (int(5000/(RATE/CHUNK)), int(10000/(RATE/CHUNK)))
            e_high = np.sum(mag[i_high[0]:i_high[1]])
            
            return rms, e_pitch, e_low, e_mid, e_high
        except:
            return 0,0,0,0,0

    def run_calibration(self):
        print("\n=== CALIBRATION (HUM CHECK) ===")
        
        def measure(name):
            print(f"\nStep: {name}")
            print("Get Ready...", end="\r"); time.sleep(1); print("GO! (Hold sound)...")
            vals = []
            st = time.time()
            while time.time()-st < 2.5: vals.append(self.get_spectrum())
            return vals

        # 1. Silence
        input("1. Silence (Enter)...")
        data = measure("Silence")
        self.silence_thresh = max(np.mean([x[0] for x in data]) * 2.0, 500)
        print(f"-> Silence Floor: {int(self.silence_thresh)}")

        # 2. OOO
        input("2. Say 'OOO' (Enter)...")
        data_o = measure("OOO")
        # Measure Pitch (Hum) for Vowels
        pitch_vals = [x[1] for x in data_o if x[0] > self.silence_thresh]
        avg_pitch_o = np.median(pitch_vals) if pitch_vals else 1000
        # Ratio O/E
        r_o = np.median([(x[3]/(x[2]+1)) for x in data_o])
        print(f"-> O Pitch: {int(avg_pitch_o)} | Ratio: {r_o:.2f}")

        # 3. EEE
        input("3. Say 'EEE' (Enter)...")
        data_e = measure("EEE")
        pitch_vals_e = [x[1] for x in data_e if x[0] > self.silence_thresh]
        avg_pitch_e = np.median(pitch_vals_e) if pitch_vals_e else 1000
        r_e = np.median([(x[3]/(x[2]+1)) for x in data_e])
        print(f"-> E Pitch: {int(avg_pitch_e)} | Ratio: {r_e:.2f}")
        
        # Set Pitch Threshold (Critical for E vs SH)
        # We set it to 40% of the average vocal pitch found
        self.pitch_thresh = min(avg_pitch_o, avg_pitch_e) * 0.4
        print(f"==> PITCH GATE: {int(self.pitch_thresh)} (Sounds below this are NOISE)")
        
        self.ratio_oe = (r_o + r_e) / 2

        # 4. SHHH
        input("4. Say 'SHHH' (Brake) (Enter)...")
        data_sh = measure("SHHH")
        r_sh = np.median([(x[4]/(x[3]+1)) for x in data_sh])
        
        # 5. SSSS
        input("5. Say 'SSSS' (Gas) (Enter)...")
        data_s = measure("SSSS")
        r_s = np.median([(x[4]/(x[3]+1)) for x in data_s])
        
        self.ratio_ssh = (r_sh + r_s) / 2
        print(f"==> S/SH Split: {self.ratio_ssh:.2f}")
        
        # 6. Clap
        input("6. CLAP (Enter)...")
        data = measure("Clap")
        self.respawn_thresh = np.max([x[0] for x in data]) * 0.8
        time.sleep(1)

    def apply(self, up, down, left, right):
        keys = [(KEY_ACCEL, up), (KEY_BRAKE, down), (KEY_LEFT, left), (KEY_RIGHT, right)]
        for k, s in keys:
            if s and not self.pressed[k]:
                pydirectinput.keyDown(k); self.pressed[k]=True
            elif not s and self.pressed[k]:
                pydirectinput.keyUp(k); self.pressed[k]=False

    def process_audio(self):
        vol, e_pitch, e_low, e_mid, e_high = self.get_spectrum()
        up, down, left, right = False, False, False, False
        status = "..."

        if vol > self.respawn_thresh:
            pydirectinput.press(KEY_RESPAWN)
            print("\r>>> RESPAWN <<<                  ", end=""); time.sleep(0.2); return

        if vol > self.silence_thresh:
            
            # --- LOGIC GATE 1: THE HUM CHECK ---
            # Is there Vocal Cord vibration?
            has_pitch = e_pitch > self.pitch_thresh

            if has_pitch:
                # IT IS A VOWEL (O or E)
                # We ignore SH/S logic completely here.
                
                # Ratio: Mid / Low
                ratio = e_mid / (e_low + 1)
                
                if ratio > self.ratio_oe:
                    left = True
                    up = True 
                    status = f"LEFT (E) [P:{int(e_pitch)}]"
                else:
                    right = True
                    up = True
                    status = f"RIGHT (O) [P:{int(e_pitch)}]"
            
            else:
                # IT IS NOISE (S or SH)
                # Pitch is low, so it must be air/hiss.
                
                # Ratio: High / Mid
                ratio = e_high / (e_mid + 1)
                
                # Force SSS if ratio is huge (safety net)
                if ratio > self.ratio_ssh or ratio > 5.0:
                    up = True
                    status = f"GAS (S) [R:{ratio:.1f}]"
                else:
                    down = True
                    status = f"BRAKE (SH) [R:{ratio:.1f}]"

            sys.stdout.write(f"\rVol:{int(vol)} | Pitch:{int(e_pitch)} | {status:<25}")
            sys.stdout.flush()

        self.apply(up, down, left, right)

    def stop(self):
        self.apply(False,False,False,False)
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

if __name__ == "__main__":
    c = VoiceController()
    c.start()