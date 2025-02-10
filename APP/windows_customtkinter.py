import os
import sys
import time
import logging
import threading
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
from motion_compression_opt import process_single_video

class TkinterLogHandler(logging.Handler):
    """
    Custom logging handler to send log records to a Tkinter Text widget.
    """
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        log_entry = self.format(record)
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, log_entry + "\n")
        self.text_widget.configure(state="disabled")
        self.text_widget.see(tk.END)

class VideoProcessingApp(ctk.CTk):
    """
    GUI application using CustomTkinter to select a video file,
    an output directory, launch video processing in a separate thread,
    show progress bars, and display total elapsed time at the end.
    The progress bars are hidden until the user starts the process.
    """
    def __init__(self):
        super().__init__()
        self.title("Video Processing (CustomTkinter)")
        self.geometry("800x600")

        ctk.set_appearance_mode("Dark")  # "Dark", "Light", or "System"
        ctk.set_default_color_theme("blue")

        self.create_widgets()
        self.configure_logging()

    def create_widgets(self):
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Configuriamo le colonne 1 e 2 con weight=1 in modo che possano espandersi
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(2, weight=1)

        # Row 0: Select input video
        self.label_file = ctk.CTkLabel(self.main_frame, text="Select a video file to process:")
        self.label_file.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.file_entry = ctk.CTkEntry(self.main_frame, width=400, placeholder_text="No file selected")
        self.file_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")

        self.btn_browse_file = ctk.CTkButton(self.main_frame, text="Browse", command=self.browse_file)
        self.btn_browse_file.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Row 1: Select output directory
        self.label_output = ctk.CTkLabel(self.main_frame, text="Select the output folder:")
        self.label_output.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.output_entry = ctk.CTkEntry(self.main_frame, width=400, placeholder_text="No folder selected")
        self.output_entry.grid(row=1, column=1, padx=5, pady=5, sticky="we")

        self.btn_browse_output = ctk.CTkButton(self.main_frame, text="Browse", command=self.browse_output)
        self.btn_browse_output.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        # Row 2: Log area
        self.log_text = tk.Text(self.main_frame, wrap="word", state="disabled", height=15)
        self.log_text.grid(row=2, column=0, columnspan=3, padx=5, pady=10, sticky="nsew")
        self.main_frame.grid_rowconfigure(2, weight=1)

        # Pre-creiamo i widget delle barre di progresso ma NON li gridiamo ancora
        self.label_progress_motion = ctk.CTkLabel(self.main_frame, text="Motion Detection Progress:")
        self.progress_bar_motion = ctk.CTkProgressBar(self.main_frame)
        self.progress_bar_motion.set(0)

        self.label_progress_comp = ctk.CTkLabel(self.main_frame, text="Compression Progress:")
        self.progress_bar_comp = ctk.CTkProgressBar(self.main_frame)
        self.progress_bar_comp.set(0)

        # Row 5: Control buttons (Start, Exit)
        self.btn_start = ctk.CTkButton(self.main_frame, text="Start", command=self.start_processing)
        self.btn_start.grid(row=5, column=1, padx=5, pady=5, sticky="e")

        self.btn_exit = ctk.CTkButton(self.main_frame, text="Exit", command=self.on_exit)
        self.btn_exit.grid(row=5, column=2, padx=5, pady=5, sticky="e")

    def configure_logging(self):
        self.log_handler = TkinterLogHandler(self.log_text)
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().setLevel(logging.DEBUG)

        if not any(isinstance(h, TkinterLogHandler) for h in logging.getLogger().handlers):
            logging.getLogger().addHandler(self.log_handler)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def browse_output(self):
        folder_path = filedialog.askdirectory(title="Select an output folder")
        if folder_path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, folder_path)

    def start_processing(self):
        file_path = self.file_entry.get().strip()
        output_dir = self.output_entry.get().strip()

        if not file_path or not os.path.isfile(file_path):
            messagebox.showerror("Error", "Please select a valid video file!")
            return
        if not output_dir or not os.path.isdir(output_dir):
            messagebox.showerror("Error", "Please select a valid output folder!")
            return

        # Disabilita il pulsante "Start" finché la procedura non è conclusa
        self.btn_start.configure(state="disabled")

        # Reset progress bars
        self.progress_bar_motion.set(0)
        self.progress_bar_comp.set(0)

        # Mostriamo le barre di avanzamento
        # - label_progress_motion in row=3, col=0
        # - progress_bar_motion in row=3, col=1 con columnspan=2 e sticky="we"
        self.label_progress_motion.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.progress_bar_motion.grid(row=3, column=1, columnspan=2, padx=5, pady=5, sticky="we")

        self.label_progress_comp.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.progress_bar_comp.grid(row=4, column=1, columnspan=2, padx=5, pady=5, sticky="we")

        self.append_log("Processing started... please wait.\n")
        logging.info("User initiated video processing.")

        # Salviamo l'istante di inizio per calcolare il tempo totale
        self.process_start_time = time.time()

        # Avviamo in un thread separato per non bloccare la GUI
        thread = threading.Thread(
            target=self.process_video_thread,
            args=(file_path, output_dir),
            daemon=True
        )
        thread.start()

    def process_video_thread(self, file_path, output_dir):
        """
        Funzione chiamata in un thread separato,
        che effettua la chiamata a process_single_video 
        e aggiorna le barre di avanzamento.
        """
        try:
            process_single_video(
                video_path=file_path,
                output_dir=output_dir,
                progress_callback_motion=self.update_progress_motion,
                progress_callback_compression=self.update_progress_compression
            )
            self.append_log("Processing completed!\n")

            # Calcolo del tempo totale
            total_time = time.time() - self.process_start_time
            self.append_log(f"Total processing time: {total_time:.2f} seconds.\n")

        except Exception as e:
            logging.error(f"Error during processing: {e}", exc_info=True)
            self.append_log(f"Error during processing: {str(e)}\n")
        finally:
            logging.info("Process completed.")
            self.enable_start_button()

    def update_progress_motion(self, value):
        """
        Callback invocato dall'elaborazione per aggiornare 
        la barra di avanzamento della motion detection.
        """
        self.after(0, lambda: self.progress_bar_motion.set(value))

    def update_progress_compression(self, value):
        """
        Callback invocato dall'elaborazione per aggiornare 
        la barra di avanzamento della compressione.
        """
        self.after(0, lambda: self.progress_bar_comp.set(value))

    def enable_start_button(self):
        self.after(0, lambda: self.btn_start.configure(state="normal"))

    def append_log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message)
        self.log_text.configure(state="disabled")
        self.log_text.see(tk.END)

    def on_exit(self):
        self.destroy()

def main():
    app = VideoProcessingApp()
    app.mainloop()

if __name__ == "__main__":
    main()
