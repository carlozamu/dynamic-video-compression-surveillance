import sys
import os
import threading
import logging
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QPlainTextEdit, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QMetaObject, Q_ARG

# Import processing functions based on the selected technique
from motion_compression_opt import process_single_video_of
from frame_differencing import process_single_video_fd

class QtLogHandler(logging.Handler, QObject):
    """Custom handler to relay log messages to the interface's text area."""
    log_signal = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

class VideoProcessingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Processing")
        self.setGeometry(100, 100, 600, 400)
        self.init_ui()
        self.setup_logging()
        self.input_files = []  # List of selected video files

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # Section: Input file selection (allowing multiple videos to be selected)
        input_layout = QHBoxLayout()
        input_label = QLabel("Input Video(s):")
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Select one or more video files")
        browse_input_btn = QPushButton("Browse")
        browse_input_btn.clicked.connect(self.browse_input)
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_line)
        input_layout.addWidget(browse_input_btn)
        layout.addLayout(input_layout)

        # Section: Output folder selection
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Folder:")
        self.output_line = QLineEdit()
        self.output_line.setPlaceholderText("Select the output folder")
        browse_output_btn = QPushButton("Browse")
        browse_output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_line)
        output_layout.addWidget(browse_output_btn)
        layout.addLayout(output_layout)

        # Section: Technique selection
        technique_layout = QHBoxLayout()
        technique_label = QLabel("Select Technique:")
        self.technique_combo = QComboBox()
        self.technique_combo.addItems(["Optical Flow", "Frame Differencing"])
        technique_layout.addWidget(technique_label)
        technique_layout.addWidget(self.technique_combo)
        layout.addLayout(technique_layout)

        # New section: Flag to perform performance analysis after conversion
        performance_layout = QHBoxLayout()
        self.performance_checkbox = QCheckBox("Perform performance analysis after conversion")
        self.performance_checkbox.setChecked(True)  # Enabled by default
        performance_layout.addWidget(self.performance_checkbox)
        layout.addLayout(performance_layout)

        # Start button
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.start_btn)

        # Log area
        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

        central_widget.setLayout(layout)

    def setup_logging(self):
        """Configure logging to send messages to the log area."""
        self.log_handler = QtLogHandler()
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.log_handler.log_signal.connect(self.append_log)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(self.log_handler)

    def append_log(self, message):
        """Append a log message in a thread-safe manner."""
        QMetaObject.invokeMethod(
            self.log_area,
            "appendPlainText",
            Qt.QueuedConnection,
            Q_ARG(str, message)
        )

    def browse_input(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_paths:
            self.input_files = file_paths
            self.input_line.setText("; ".join(file_paths))

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_line.setText(folder)

    def start_processing(self):
        if not self.input_files:
            QMessageBox.critical(self, "Error", "Please select at least one input video file!")
            return

        output_folder = self.output_line.text().strip()
        if not output_folder or not os.path.isdir(output_folder):
            QMessageBox.critical(self, "Error", "Please select a valid output folder!")
            return

        self.append_log("Starting processing of selected videos...")
        # Disable the button to prevent multiple executions
        QMetaObject.invokeMethod(self.start_btn, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, False))
        technique = self.technique_combo.currentText()
        # Capture the state of the checkbox before initiating the thread
        run_performance = self.performance_checkbox.isChecked()

        def process_videos():
            # Process the selected videos
            for input_path in self.input_files:
                if not os.path.isfile(input_path):
                    logging.error(f"File not found: {input_path}")
                    continue
                try:
                    if technique == "Optical Flow":
                        logging.info(f"Processing {input_path} using Optical Flow.")
                        process_single_video_of(input_path, output_folder)
                    elif technique == "Frame Differencing":
                        logging.info(f"Processing {input_path} using Frame Differencing.")
                        process_single_video_fd(input_path, output_folder)
                    else:
                        logging.error("Unknown technique selected.")
                except Exception as e:
                    logging.error(f"Error processing {input_path}: {e}", exc_info=True)

            logging.info("All selected videos processed.")
            self.append_log("Processing completed.")

            # Execute performance analysis based on the selected flag
            if run_performance:
                logging.info("Starting performance analysis.")
                self.append_log("Starting performance analysis.")
                try:
                    # Construct the path to the performance_analysis.py script
                    performance_script = os.path.join(os.path.dirname(__file__), "performance_analysis.py")
                    # Run the script, passing the output folder as an argument
                    result = subprocess.run(
                        ["python", performance_script, output_folder],
                        capture_output=True, text=True
                    )
                    if result.returncode != 0:
                        logging.error("Performance analysis failed: " + result.stderr)
                        self.append_log("Performance analysis failed.")
                    else:
                        logging.info("Performance analysis completed successfully.")
                        self.append_log("Performance analysis completed successfully.")
                        log_msg = f"Performance files saved in: {output_folder}"
                        logging.info(log_msg)
                        self.append_log(log_msg)
                except Exception as e:
                    logging.error("Error running performance analysis: " + str(e), exc_info=True)
                    self.append_log("Error running performance analysis.")
            else:
                logging.info("Skipping performance analysis as per user choice.")
                self.append_log("Skipping performance analysis.")

            # Re-enable the Start button
            QMetaObject.invokeMethod(self.start_btn, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, True))

        # Start processing in a separate thread to avoid blocking the GUI
        thread = threading.Thread(target=process_videos)
        thread.start()

def main():
    app = QApplication(sys.argv)
    window = VideoProcessingWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()