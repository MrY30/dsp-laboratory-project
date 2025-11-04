import sys
import numpy as np
import sounddevice as sd
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer
import pyqtgraph as pg

# --- Constants ---
SAMPLERATE = 44100      # Samples per second (standard audio rate)
CHUNKSIZE = 1024        # Number of samples to read at a time
APP_TITLE = "Real-Time Audio Waveform"

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
        # This is the line object we will update
        self.data_line = self.plot_widget.plot(pen='c') # 'c' for cyan

        # 4. --- Set up the audio stream ---
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLERATE,
                channels=1,         # Mono audio
                blocksize=CHUNKSIZE,
                dtype='float32'     # Data type of the samples
            )
            self.stream.start()
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.close()

        # 5. --- Set up the update timer ---
        # We use a QTimer to repeatedly call our update function
        self.timer = QTimer()
        self.timer.setInterval(30) # Refresh rate in milliseconds (approx. 33 FPS)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        """
        This function is called by the QTimer.
        It reads new data from the audio stream and updates the plot.
        """
        try:
            # Read a chunk of data from the audio stream
            # 'data' is a numpy array. 'overflowed' is a boolean.
            data, overflowed = self.stream.read(CHUNKSIZE)
            
            if overflowed:
                print("Warning: Audio buffer overflowed")

            # 'data' has a shape of (CHUNKSIZE, 1). We 'flatten' it to 1D.
            self.data_line.setData(data.flatten())

        except Exception as e:
            print(f"Error during audio read or plot update: {e}")

    def closeEvent(self, event):
        """
        This function is called automatically when the window is closed.
        We need to stop the timer and close the audio stream.
        """
        self.timer.stop()
        self.stream.stop()
        self.stream.close()
        event.accept() # Accept the close event

# --- Main execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioMonitorWindow()
    window.show()
    sys.exit(app.exec())