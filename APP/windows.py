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
    QTextEdit,
    QWidget,
    QMessageBox,
)
from PyQt5.QtCore import pyqtSignal, QObject
from motion_compression import process_single_video

class LogHandler(logging.Handler, QObject):
    log_signal = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        log_message = self.format(record)
        self.log_signal.emit(log_message)

class VideoProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Elaborazione Video")
        self.setGeometry(100, 100, 800, 600)

        self.init_ui()

        # Crea il tuo handler personalizzato
        self.log_handler = LogHandler()
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.log_handler.log_signal.connect(self.append_log)

        # Configura il logger di root
        root_logger = logging.getLogger()
        # Imposta il livello di log: DEBUG per mostrare tutto (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        root_logger.setLevel(logging.DEBUG)

        # (Opzionale) Per continuare a vedere i log anche in console/terminale
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Rimuovi eventuali handler di default per evitare duplicati
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Aggiungi i nuovi handler al root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(self.log_handler)

    def init_ui(self):
        # Layout principale
        main_layout = QVBoxLayout()

        # Selezione file video
        file_layout = QHBoxLayout()
        file_label = QLabel("Seleziona un file video da elaborare:")
        self.file_input = QLineEdit()
        self.file_input.setReadOnly(True)
        file_button = QPushButton("Sfoglia")
        file_button.clicked.connect(self.browse_file)
        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(file_button)
        main_layout.addLayout(file_layout)

        # Selezione cartella output
        output_layout = QHBoxLayout()
        output_label = QLabel("Seleziona la cartella di output:")
        self.output_input = QLineEdit()
        self.output_input.setReadOnly(True)
        output_button = QPushButton("Sfoglia")
        output_button.clicked.connect(self.browse_output)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_input)
        output_layout.addWidget(output_button)
        main_layout.addLayout(output_layout)

        # Area di testo per i log
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        main_layout.addWidget(self.log_area)

        # Pulsanti
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Avvia")
        self.start_button.clicked.connect(self.start_processing)
        self.exit_button = QPushButton("Esci")
        self.exit_button.clicked.connect(self.close)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.exit_button)
        main_layout.addLayout(button_layout)

        # Widget principale
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleziona un file video", "", "Video Files (*.mp4)")
        if file_path:
            self.file_input.setText(file_path)

    def browse_output(self):
        output_dir = QFileDialog.getExistingDirectory(self, "Seleziona una cartella di output")
        if output_dir:
            self.output_input.setText(output_dir)

    def start_processing(self):
        file_path = self.file_input.text()
        output_dir = self.output_input.text()

        if not file_path or not os.path.exists(file_path):
            QMessageBox.critical(self, "Errore", "Seleziona un file video valido!")
            return
        if not output_dir or not os.path.exists(output_dir):
            QMessageBox.critical(self, "Errore", "Seleziona una cartella di output valida!")
            return

        self.log_area.append("Elaborazione avviata... attendere il completamento.\n")
        logging.info("Iniziato il thread di elaborazione del video.")

        # Avvia l'elaborazione in un thread separato
        threading.Thread(
            target=self.process_video,
            args=(file_path, output_dir),
            daemon=True
        ).start()

    def process_video(self, file_path, output_dir):
        try:
            logging.debug(f"Chiamata a process_single_video con file: {file_path}, cartella di output: {output_dir}")
            process_single_video(file_path, output_dir)
            self.log_area.append("Elaborazione completata!")
        except Exception as e:
            logging.error(f"Errore durante l'elaborazione: {e}", exc_info=True)
        finally:
            logging.info("Processo completato.")

    def append_log(self, message):
        self.log_area.append(message)

def main():
    app = QApplication(sys.argv)
    window = VideoProcessingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()