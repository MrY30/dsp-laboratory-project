import sys
import numpy as np
import sounddevice as sd
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer
import pyqtgraph as pg
# --- NEW IMPORT ---
from scipy.signal import butter, lfilter, lfilter_zi

# --- Constants ---
SAMPLERATE = 44100      # Samples per second
CHUNKSIZE = 1024        # Samples per chunk
APP_TITLE = "Real-Time Audio Waveform (with Filter)"

# --- NEW FILTER CONSTANTS ---
FILTER_ORDER = 4        # Order of the Butterworth filter
CUTOFF_FREQ = 80.0      # Cutoff frequency in Hz
FILTER_TYPE = 'highpass'

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
        self.plot_widget.setYRange(-0.5, 0.5, padding=0)
        self.plot_widget.setXRange(0, CHUNKSIZE, padding=0)
        # --- NEW: Add a legend ---
        self.plot_widget.addLegend(offset=(10, 10))

        # 3. --- Create the plot data lines ---
        # --- MODIFIED: Renamed to data_line_raw ---
        self.data_line_raw = self.plot_widget.plot(
            pen='c', name='Raw Signal'
        )
        # --- NEW: Added a line for the filtered signal ---
        self.data_line_filtered = self.plot_widget.plot(
            pen='y', name='Filtered Signal'
        )

        # 4. --- NEW: Design the filter ---
        # Nyquist frequency is half the sample rate
        nyquist = 0.5 * SAMPLERATE
        # Normalize the cutoff frequency
        normal_cutoff = CUTOFF_FREQ / nyquist
        # Design the filter and get coefficients (b, a)
        self.b, self.a = butter(
            FILTER_ORDER, normal_cutoff, btype=FILTER_TYPE, analog=False
        )
        # Get the initial filter state (the "memory")
        self.zi = lfilter_zi(self.b, self.a)
        
        # 5. --- Set up the audio stream ---
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLERATE,
                channels=1,
                blocksize=CHUNKSIZE,
                dtype='float32'
            )
            self.stream.start()
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.close()

        # 6. --- Set up the update timer ---
        self.timer = QTimer()
        self.timer.setInterval(30) # Refresh rate in milliseconds
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        """
        This function is called by the QTimer.
        It reads, filters, and plots the new audio data.
        """
        try:
            # Read a chunk of data from the audio stream
            data, overflowed = self.stream.read(CHUNKSIZE)
            
            if overflowed:
                print("Warning: Audio buffer overflowed")

            # Get the raw data (flattened to 1D)
            raw_data = data.flatten()

            # --- NEW: Apply the filter ---
            # Pass in the data and the filter's previous state (self.zi)
            # Get back the filtered data and the new state (which we save)
            filtered_data, self.zi = lfilter(
                self.b, self.a, raw_data, zi=self.zi
            )

            # --- MODIFIED: Update both plot lines ---
            self.data_line_raw.setData(raw_data)
            self.data_line_filtered.setData(filtered_data)

        except Exception as e:
            print(f"Error during audio read or plot update: {e}")

    def closeEvent(self, event):
        """
        Clean up when the window is closed.
        """
        self.timer.stop()
        self.stream.stop()
        self.stream.close()
        event.accept()

# --- Main execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioMonitorWindow()
    window.show()
    sys.exit(app.exec())