import sys
import os
import threading
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QPlainTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QMetaObject, Q_ARG

# Importa le funzioni di processing in base alla tecnica
from motion_compression_opt import process_single_video_of
from frame_differencing import process_single_video_fd

class QtLogHandler(logging.Handler, QObject):
    """Handler personalizzato per inviare i log all'area di testo dell'interfaccia."""
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

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # Sezione: selezione file di input
        input_layout = QHBoxLayout()
        input_label = QLabel("Input Video:")
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Seleziona il file video")
        browse_input_btn = QPushButton("Browse")
        browse_input_btn.clicked.connect(self.browse_input)
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_line)
        input_layout.addWidget(browse_input_btn)
        layout.addLayout(input_layout)

        # Sezione: selezione cartella di output
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Folder:")
        self.output_line = QLineEdit()
        self.output_line.setPlaceholderText("Seleziona la cartella di output")
        browse_output_btn = QPushButton("Browse")
        browse_output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_line)
        output_layout.addWidget(browse_output_btn)
        layout.addLayout(output_layout)

        # Sezione: selezione tecnica
        technique_layout = QHBoxLayout()
        technique_label = QLabel("Select Technique:")
        self.technique_combo = QComboBox()
        self.technique_combo.addItems(["Optical Flow", "Frame Differencing"])
        technique_layout.addWidget(technique_label)
        technique_layout.addWidget(self.technique_combo)
        layout.addLayout(technique_layout)

        # Pulsante di avvio
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.start_btn)

        # Area di log
        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

        central_widget.setLayout(layout)

    def setup_logging(self):
        """Configura il logging per inviare i messaggi all’area di log."""
        self.log_handler = QtLogHandler()
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.log_handler.log_signal.connect(self.append_log)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(self.log_handler)

    def append_log(self, message):
        """Aggiunge un messaggio di log in maniera thread-safe."""
        QMetaObject.invokeMethod(
            self.log_area,
            "appendPlainText",
            Qt.QueuedConnection,
            Q_ARG(str, message)
        )

    def browse_input(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.input_line.setText(file_path)

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_line.setText(folder)

    def start_processing(self):
        input_path = self.input_line.text().strip()
        output_folder = self.output_line.text().strip()
        technique = self.technique_combo.currentText()

        if not input_path or not os.path.isfile(input_path):
            QMessageBox.critical(self, "Error", "Please select a valid input video file!")
            return
        if not output_folder or not os.path.isdir(output_folder):
            QMessageBox.critical(self, "Error", "Please select a valid output folder!")
            return

        self.append_log("Starting processing...")
        self.start_btn.setEnabled(False)
        # Avvia il processing in un thread separato
        thread = threading.Thread(target=self.process_video, args=(input_path, output_folder, technique))
        thread.start()

    def process_video(self, input_path, output_folder, technique):
        try:
            if technique == "Optical Flow":
                logging.info("Technique: Optical Flow selected.")
                process_single_video_of(input_path, output_folder)
            elif technique == "Frame Differencing":
                logging.info("Technique: Frame Differencing selected.")
                process_single_video_fd(input_path, output_folder)
            else:
                logging.error("Unknown technique selected.")
                return
            logging.info("Processing completed successfully.")
        except Exception as e:
            logging.error(f"Error during processing: {e}", exc_info=True)
        finally:
            self.start_btn.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    window = VideoProcessingWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()