import sys
import numpy as np
import sounddevice as sd
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer
import pyqtgraph as pg

# --- Constants ---
SAMPLERATE = 44100      # Samples per second (standard audio rate)
CHUNKSIZE = 1024        # Number of samples to read at a time
APP_TITLE = "Real-Time Audio Waveform & Stats"

class AudioMonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 1. --- Set up the main window ---
        self.setWindowTitle(APP_TITLE)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 2. --- Create the pyqtgraph plot ---
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Configure plot appearance
        self.plot_widget.setTitle("Live Audio Input")
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Samples')
        self.plot_widget.setYRange(-0.5, 0.5, padding=0) # Audio is float32, -1 to 1
        self.plot_widget.setXRange(0, CHUNKSIZE, padding=0)

        # 3. --- Create the plot data line ---
        self.data_line = self.plot_widget.plot(pen='c') # 'c' for cyan

        # 4. --- Set up the audio stream ---
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLERATE,
                channels=1,        # Mono audio
                blocksize=CHUNKSIZE,
                dtype='float32'    # Data type of the samples
            )
            self.stream.start()
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.close()
            return # --- NEW --- Stop initialization if stream fails

        # 5. --- Set up the update timer ---
        self.timer = QTimer()
        self.timer.setInterval(30) # Refresh rate in milliseconds (approx. 33 FPS)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        """
        This function is called by the QTimer.
        It reads new data from the audio stream, updates the plot,
        and prints loudness and pitch to the console.
        """
        try:
            # Read a chunk of data from the audio stream
            data, overflowed = self.stream.read(CHUNKSIZE)
            
            if overflowed:
                print("Warning: Audio buffer overflowed")

            # 'data' has a shape of (CHUNKSIZE, 1). We 'flatten' it to 1D.
            data_1d = data.flatten()
            
            # Update the plot
            self.data_line.setData(data_1d)

            # --- NEW: Calculate Loudness (RMS Amplitude) ---
            # Square all values, get the mean, then take the square root.
            rms_amplitude = np.sqrt(np.mean(np.square(data_1d)))

            # --- NEW: Calculate Pitch (Fundamental Frequency via FFT) ---
            # Perform Fast Fourier Transform
            fft_spectrum = np.abs(np.fft.rfft(data_1d))
            
            # Get the frequencies corresponding to the FFT bins
            # 1.0 / SAMPLERATE is the sample spacing
            fft_freqs = np.fft.rfftfreq(len(data_1d), 1.0 / SAMPLERATE)

            # Find the peak frequency (ignoring the 0Hz DC offset)
            peak_index = np.argmax(fft_spectrum[1:]) + 1
            pitch_hz = fft_freqs[peak_index]
            peak_magnitude = fft_spectrum[peak_index]

            # --- NEW: Simple Voicing Detection ---
            # If the signal is very quiet or the peak isn't prominent,
            # it's likely unvoiced (noise/silence).
            if rms_amplitude < 0.005 or peak_magnitude < np.mean(fft_spectrum[1:]) * 10:
                display_pitch = "--- (unvoiced)"
            else:
                display_pitch = f"{pitch_hz:7.1f} Hz"

            # --- NEW: Print to Console ---
            # Use carriage return '\r' to print on the same line
            print(f"Loudness (RMS): {rms_amplitude:.4f}  |  Pitch: {display_pitch}      ", end='\r')


        except Exception as e:
            # Check if the error is due to a closed stream
            if "Stream is stopped" in str(e) or "Stream is closed" in str(e):
                print("\nAudio stream closed.")
                self.timer.stop() # Stop the timer if stream is dead
            else:
                print(f"\nError during audio read or plot update: {e}")

    def closeEvent(self, event):
        """
        This function is called automatically when the window is closed.
        We need to stop the timer and close the audio stream.
        """
        print("\nClosing application...")
        self.timer.stop()
        
        # --- NEW: Check if stream exists before closing ---
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
            
        event.accept() # Accept the close event

# --- Main execution ---
if __name__ == "__main__":
    # --- NEW: Ensure Qt gracefully handles exceptions ---
    try:
        app = QApplication(sys.argv)
        window = AudioMonitorWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)
