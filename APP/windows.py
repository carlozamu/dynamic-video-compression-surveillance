import sys
import os
import logging
import threading
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QPlainTextEdit,
    QWidget,
    QMessageBox,
)
from PyQt5.QtCore import pyqtSignal, QObject, QMetaObject, Qt, Q_ARG
from motion_compression_opt import process_single_video

class LogHandler(logging.Handler, QObject):
    """
    Custom logging handler that emits logs through a Qt signal
    so they can be displayed in the GUI.
    """
    log_signal = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        """
        When a log record is emitted, convert it to text and
        send it through the signal.
        """
        log_message = self.format(record)
        self.log_signal.emit(log_message)

class VideoProcessingApp(QMainWindow):
    """
    Main window of the application. It contains:
      - A file selector for the input video.
      - A directory selector for the output folder.
      - A text area for logs.
      - Buttons to start and exit the processing.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Processing")
        self.setGeometry(100, 100, 800, 600)

        self.init_ui()

        self.log_handler = LogHandler()
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.log_handler.log_signal.connect(self.append_log)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        if not any(isinstance(handler, LogHandler) for handler in root_logger.handlers):
            root_logger.addHandler(console_handler)
            root_logger.addHandler(self.log_handler)

    def init_ui(self):
        """
        Builds the layout and widgets of the main window. 
        Includes text fields, buttons, and the log display area.
        """
        main_layout = QVBoxLayout()

        file_layout = QHBoxLayout()
        file_label = QLabel("Select a video file to process:")
        self.file_input = QLineEdit()
        self.file_input.setReadOnly(True)
        file_button = QPushButton("Browse")
        file_button.clicked.connect(self.browse_file)
        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(file_button)
        main_layout.addLayout(file_layout)

        output_layout = QHBoxLayout()
        output_label = QLabel("Select the output folder:")
        self.output_input = QLineEdit()
        self.output_input.setReadOnly(True)
        output_button = QPushButton("Browse")
        output_button.clicked.connect(self.browse_output)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_input)
        output_layout.addWidget(output_button)
        main_layout.addLayout(output_layout)

        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)
        main_layout.addWidget(self.log_area)

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_processing)
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.exit_button)
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def browse_file(self):
        """
        Opens a file dialog to select a video file (.mp4).
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a video file", "", "Video Files (*.mp4)")
        if file_path:
            self.file_input.setText(file_path)

    def browse_output(self):
        """
        Opens a folder selection dialog to choose the output directory.
        """
        output_dir = QFileDialog.getExistingDirectory(self, "Select an output folder")
        if output_dir:
            self.output_input.setText(output_dir)

    def start_processing(self):
        """
        Validates the input file and output directory, then starts the 
        video processing in a separate thread to keep the UI responsive.
        """
        file_path = self.file_input.text()
        output_dir = self.output_input.text()

        if not file_path or not os.path.exists(file_path):
            QMessageBox.critical(self, "Error", "Please select a valid video file!")
            return
        if not output_dir or not os.path.exists(output_dir):
            QMessageBox.critical(self, "Error", "Please select a valid output folder!")
            return

        self.log_area.appendPlainText("Processing started... please wait for completion.\n")
        logging.info("Thread for video processing started.")

        threading.Thread(
            target=self.process_video,
            args=(file_path, output_dir),
            daemon=True
        ).start()

    def process_video(self, file_path, output_dir):
        """
        Calls the function 'process_single_video' to do the actual processing.
        Catches exceptions and logs them if needed.
        """
        try:
            logging.debug(f"Calling process_single_video with file: {file_path}, output folder: {output_dir}")
            process_single_video(file_path, output_dir)
            self.append_log("Processing completed!")
        except Exception as e:
            logging.error(f"Error during processing: {e}", exc_info=True)
            self.append_log(f"Error during processing: {str(e)}")
        finally:
            logging.info("Process completed.")

    def append_log(self, message):
        """
        Appends a log message to the log_area (in a thread-safe manner).
        """
        QMetaObject.invokeMethod(
            self.log_area,
            "appendPlainText",
            Qt.QueuedConnection,
            Q_ARG(str, message)
        )

def main():
    """
    Main function that initializes the QApplication and shows the main window.
    """
    app = QApplication(sys.argv)
    window = VideoProcessingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
