import PySimpleGUI as sg
import os
import logging
import threading
from motion_compression import process_single_video

class GuiLogger(logging.Handler):
    """
    Un handler di logging che inoltra ogni messaggio di log a PySimpleGUI,
    permettendo di visualizzare i log nella finestra.
    """
    def __init__(self, window, key):
        super().__init__()
        self.window = window
        self.key = key

    def emit(self, record):
        log_message = self.format(record)
        # Inoltra il messaggio di log alla GUI come evento "-LOG_EVENT-"
        self.window.write_event_value(self.key, log_message)

def setup_logging(window, key):
    """
    Configura il logger principale di Python perché invii i messaggi
    anche alla finestra PySimpleGUI, tramite GuiLogger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = GuiLogger(window, key)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def process_in_thread(file_path, output_dir, window):
    """
    Funzione eseguita nel thread: chiama la funzione di elaborazione
    e poi notifica la GUI al termine.
    """
    try:
        process_single_video(file_path, output_dir)
    finally:
        # Al termine dell'elaborazione, invia un segnale alla GUI
        window.write_event_value("-PROCESS_DONE-", "Elaborazione completata!")

# Layout dell'interfaccia
layout = [
    [sg.Text("Seleziona un file video da elaborare:")],
    [sg.Input(key="-FILE-", enable_events=True, visible=True), 
     sg.FileBrowse(file_types=(("Video Files", "*.mp4"),))],
    [sg.Text("Seleziona la cartella di output:")],
    [sg.Input(key="-OUTPUT-", enable_events=True, visible=True), sg.FolderBrowse()],
    [sg.Multiline(size=(60, 20), key="-LOG-", disabled=True, autoscroll=True)],
    [sg.Button("Avvia"), sg.Button("Esci")]
]

# Crea la finestra
window = sg.Window("Elaborazione Video", layout, finalize=True)

# Collega il logger alla GUI
setup_logging(window, "-LOG_EVENT-")

while True:
    event, values = window.read()
    
    if event in (sg.WINDOW_CLOSED, "Esci"):
        break

    if event == "Avvia":
        file_path = values["-FILE-"]
        output_dir = values["-OUTPUT-"]

        if not file_path or not os.path.exists(file_path):
            sg.popup_error("Seleziona un file video valido!")
        elif not output_dir or not os.path.exists(output_dir):
            sg.popup_error("Seleziona una cartella di output valida!")
        else:
            # Stampa un messaggio nell'area di log anziché aprire un popup
            window["-LOG-"].print("Elaborazione avviata... attendere il completamento.\n")
            
            # Avvia l'elaborazione in un thread per non bloccare la GUI
            threading.Thread(
                target=process_in_thread,
                args=(file_path, output_dir, window),
                daemon=True
            ).start()

    # Questo evento riceve i messaggi di log e li mostra nella Multiline
    elif event == "-LOG_EVENT-":
        log_message = values["-LOG_EVENT-"]
        window["-LOG-"].print(log_message)
    
    # Evento che segnala la fine dell'elaborazione
    elif event == "-PROCESS_DONE-":
        window["-LOG-"].print(values["-PROCESS_DONE-"])
        # Puoi anche mostrare un popup se vuoi evidenziare che è finita
        # sg.popup("Elaborazione completata!")

window.close()