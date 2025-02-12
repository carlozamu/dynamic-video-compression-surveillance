#!/usr/bin/env python
import os
import sys
import re
import csv
import matplotlib.pyplot as plt
import cv2  # Necessario per leggere le proprietà dei video

def parse_execution_times(file_path):
    """
    Estrae dal file execution_times.txt i dati di performance.
    
    Per il formato Optical Flow (execution_times.txt prodotto da motion_compression_opt.py)
      Motion Detection:
        Frames processed: <md_frames>
        Total time: <md_time> seconds
        Average time per frame: <md_avg> seconds

      Compression:
        Frames processed: <cp_frames>
        Total time: <cp_time> seconds
        Average time per frame: <cp_avg> seconds

      Total video processing time: <total_processing_time> seconds

    Per il formato Frame Differencing (execution_times.txt prodotto da frame_differencing.py)
      Frame Differencing:
        Frames processed: <frames>
        Total time: <time> seconds
        Average time per frame: <avg> seconds

      Total video processing time: <total_processing_time> seconds

    Nel caso del Frame Differencing, i valori verranno assegnati alle chiavi md_*
    e i valori relativi alla compressione (cp_*) saranno impostati a 0.
    """
    data = {}
    try:
        with open(file_path, "r") as f:
            # Legge le linee ignorando quelle vuote
            lines = [line.strip() for line in f if line.strip() != '']
        pattern = r":\s*([\d\.]+)"
        
        if lines[0].startswith("Motion Detection:"):
            # Parsing Optical Flow:
            # Expected lines:
            #   0: Motion Detection:
            #   1: Frames processed: <md_frames>
            #   2: Total time: <md_time> seconds
            #   3: Average time per frame: <md_avg> seconds
            md_frames = int(re.search(pattern, lines[1]).group(1))
            md_time   = float(re.search(pattern, lines[2]).group(1))
            md_avg    = float(re.search(pattern, lines[3]).group(1))
            # Trova la sezione Compression:
            comp_index = None
            for i, line in enumerate(lines):
                if line.startswith("Compression:"):
                    comp_index = i
                    break
            if comp_index is not None:
                cp_frames = int(re.search(pattern, lines[comp_index+1]).group(1))
                cp_time   = float(re.search(pattern, lines[comp_index+2]).group(1))
                cp_avg    = float(re.search(pattern, lines[comp_index+3]).group(1))
            else:
                cp_frames = cp_time = cp_avg = 0
            # Trova la riga del tempo totale
            total_line = [line for line in lines if line.startswith("Total video processing time:")]
            if total_line:
                total_processing_time = float(re.search(pattern, total_line[0]).group(1))
            else:
                total_processing_time = md_time + cp_time

            data = {
                "md_frames": md_frames,
                "md_time": md_time,
                "md_avg": md_avg,
                "cp_frames": cp_frames,
                "cp_time": cp_time,
                "cp_avg": cp_avg,
                "total_processing_time": total_processing_time
            }
        elif lines[0].startswith("Frame Differencing:"):
            # Parsing Frame Differencing:
            # Expected lines:
            #   0: Frame Differencing:
            #   1: Frames processed: <frames>
            #   2: Total time: <time> seconds
            #   3: Average time per frame: <avg> seconds
            fd_frames = int(re.search(pattern, lines[1]).group(1))
            fd_time   = float(re.search(pattern, lines[2]).group(1))
            fd_avg    = float(re.search(pattern, lines[3]).group(1))
            total_line = [line for line in lines if line.startswith("Total video processing time:")]
            if total_line:
                total_processing_time = float(re.search(pattern, total_line[0]).group(1))
            else:
                total_processing_time = fd_time

            # Assegna i valori del Frame Differencing alle chiavi md_*
            data = {
                "md_frames": fd_frames,
                "md_time": fd_time,
                "md_avg": fd_avg,
                "cp_frames": 0,
                "cp_time": 0.0,
                "cp_avg": 0.0,
                "total_processing_time": total_processing_time
            }
        else:
            raise ValueError("Formato non riconosciuto di execution_times.txt")
    except Exception as e:
        print(f"Errore nel parsing di {file_path}: {e}")
        return None
    return data

def get_video_duration(video_path):
    """Restituisce la durata del video in secondi utilizzando cv2."""
    duration = 0
    if os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0:
                duration = frame_count / fps
            cap.release()
    return duration

def get_file_size(file_path):
    """Restituisce la dimensione del file in byte."""
    try:
        return os.path.getsize(file_path)
    except Exception:
        return 0

def get_original_and_compressed_paths(subfolder):
    """
    Restituisce una tuple (original_path, compressed_path) in base alla presenza
    dei file nelle sottocartelle. Se esiste 'overlay.mp4' e 'compressed.mp4', li usa;
    altrimenti prova con 'dilated_motion_mask_video.mp4' e 'compressed_final_video.mp4'.
    """
    overlay_path = os.path.join(subfolder, "overlay.mp4")
    compressed_path = os.path.join(subfolder, "compressed.mp4")
    if os.path.isfile(overlay_path) and os.path.isfile(compressed_path):
        return overlay_path, compressed_path
    else:
        original_path = os.path.join(subfolder, "dilated_motion_mask_video.mp4")
        compressed_path = os.path.join(subfolder, "compressed_final_video.mp4")
        if os.path.isfile(original_path) and os.path.isfile(compressed_path):
            return original_path, compressed_path
    return None, None

def main():
    if len(sys.argv) < 2:
        print("Uso: python performance_analysis.py <output_folder>")
        sys.exit(1)
        
    output_folder = sys.argv[1]
    if not os.path.isdir(output_folder):
        print(f"Cartella di output non valida: {output_folder}")
        sys.exit(1)
    
    # Creazione della sottocartella 'performance'
    performance_folder = os.path.join(output_folder, "performance")
    os.makedirs(performance_folder, exist_ok=True)
    
    performance_data = []
    
    # Scorriamo tutte le sottocartelle della cartella di output
    for item in os.listdir(output_folder):
        subfolder = os.path.join(output_folder, item)
        if os.path.isdir(subfolder):
            exec_file = os.path.join(subfolder, "execution_times.txt")
            if os.path.isfile(exec_file):
                data = parse_execution_times(exec_file)
                if data is not None:
                    data["video"] = item  # Nome della sottocartella (video)
                    
                    # Ottiene i percorsi per il video "originale" e quello "compresso"
                    original_path, compressed_path = get_original_and_compressed_paths(subfolder)
                    if original_path is None or compressed_path is None:
                        print(f"Attenzione: file video non trovati in {subfolder}")
                        continue
                    
                    # Legge la durata del video originale
                    video_duration = get_video_duration(original_path)
                    data["video_duration_seconds"] = video_duration
                    
                    # Calcola il tempo di conversione medio per minuto di video
                    if video_duration > 0:
                        data["conversion_time_per_minute"] = data["total_processing_time"] * 60 / video_duration
                    else:
                        data["conversion_time_per_minute"] = 0
                        
                    # Calcola dimensioni originali e compresso
                    original_size = get_file_size(original_path)
                    compressed_size = get_file_size(compressed_path)
                    data["original_size_bytes"] = original_size
                    data["compressed_size_bytes"] = compressed_size
                    
                    if original_size > 0:
                        reduction_percentage = (original_size - compressed_size) / original_size * 100
                    else:
                        reduction_percentage = 0
                    data["reduction_percentage"] = reduction_percentage
                    
                    performance_data.append(data)
    
    if not performance_data:
        print("Nessun dato di performance trovato.")
        sys.exit(1)
    
    # Definisce i nomi estesi per il CSV
    fieldnames = [
        "video",
        "md_frames",
        "md_time (s)",
        "md_avg (s/frame)",
        "cp_frames",
        "cp_time (s)",
        "cp_avg (s/frame)",
        "total_processing_time (s)",
        "video_duration_seconds",
        "conversion_time_per_minute (s/min)",
        "original_size_bytes",
        "compressed_size_bytes",
        "reduction_percentage (%)"
    ]
    
    csv_file = os.path.join(performance_folder, "performance_data.csv")
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in performance_data:
            writer.writerow({
                "video": d.get("video", ""),
                "md_frames": d.get("md_frames", ""),
                "md_time (s)": d.get("md_time", ""),
                "md_avg (s/frame)": d.get("md_avg", ""),
                "cp_frames": d.get("cp_frames", ""),
                "cp_time (s)": d.get("cp_time", ""),
                "cp_avg (s/frame)": d.get("cp_avg", ""),
                "total_processing_time (s)": d.get("total_processing_time", ""),
                "video_duration_seconds": d.get("video_duration_seconds", ""),
                "conversion_time_per_minute (s/min)": d.get("conversion_time_per_minute", ""),
                "original_size_bytes": d.get("original_size_bytes", ""),
                "compressed_size_bytes": d.get("compressed_size_bytes", ""),
                "reduction_percentage (%)": d.get("reduction_percentage", "")
            })
    print(f"CSV salvato in: {csv_file}")
    
    # Grafico lineare: Tempo di conversione totale e conversione per minuto per ogni video
    videos = [d["video"] for d in performance_data]
    total_times = [d["total_processing_time"] for d in performance_data]
    conv_time_per_min = [d["conversion_time_per_minute"] for d in performance_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(videos, total_times, marker='o', label="Tempo Conversione Totale (s)")
    plt.plot(videos, conv_time_per_min, marker='o', label="Tempo di Conversione per Minuto (s/min)")
    plt.xlabel("Video")
    plt.ylabel("Tempo (s)")
    plt.title("Tempo di Conversione Totale e per Minuto per Video")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    line_chart_path = os.path.join(performance_folder, "conversion_times_line_chart.png")
    plt.savefig(line_chart_path)
    plt.close()
    print(f"Grafico lineare salvato in: {line_chart_path}")
    
    # Grafico a barre: Percentuale di riduzione per ogni video con media complessiva
    reduction_percentages = [d["reduction_percentage"] for d in performance_data]
    avg_reduction = sum(reduction_percentages) / len(reduction_percentages)
    
    plt.figure(figsize=(10, 6))
    plt.bar(videos, reduction_percentages, color="cornflowerblue", label="Riduzione (%)")
    plt.axhline(y=avg_reduction, color="red", linestyle="--", label=f"Media Riduzione ({avg_reduction:.2f}%)")
    plt.xlabel("Video")
    plt.ylabel("Riduzione (%)")
    plt.title("Percentuale di Compressione per Video")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    bar_chart_path = os.path.join(performance_folder, "reduction_percentage_bar_chart.png")
    plt.savefig(bar_chart_path)
    plt.close()
    print(f"Grafico a barre salvato in: {bar_chart_path}")
    
    print("Analisi delle performance completata con successo.")

if __name__ == "__main__":
    main()