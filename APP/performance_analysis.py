#!/usr/bin/env python
import os
import sys
import re
import csv
import matplotlib.pyplot as plt
import cv2  # Necessario per leggere le proprietà dei video

def parse_execution_times(file_path):
    """
    Estrae dal file execution_times.txt i seguenti dati:
      - md_frames, md_time, md_avg
      - cp_frames, cp_time, cp_avg
      - total_processing_time

    Formato atteso:
    
      Motion Detection:
        Frames processed: <md_frames>
        Total time: <md_time> seconds
        Average time per frame: <md_avg> seconds

      Compression:
        Frames processed: <cp_frames>
        Total time: <cp_time> seconds
        Average time per frame: <cp_avg> seconds

      Total video processing time: <total_processing_time> seconds
    """
    data = {}
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        pattern = r":\s*([\d\.]+)"
        md_frames = int(re.search(pattern, lines[1]).group(1))
        md_time   = float(re.search(pattern, lines[2]).group(1))
        md_avg    = float(re.search(pattern, lines[3]).group(1))
        cp_frames = int(re.search(pattern, lines[6]).group(1))
        cp_time   = float(re.search(pattern, lines[7]).group(1))
        cp_avg    = float(re.search(pattern, lines[8]).group(1))
        total_processing_time = float(re.search(pattern, lines[10]).group(1))
        
        data = {
            "md_frames": md_frames,
            "md_time": md_time,
            "md_avg": md_avg,
            "cp_frames": cp_frames,
            "cp_time": cp_time,
            "cp_avg": cp_avg,
            "total_processing_time": total_processing_time
        }
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
                    
                    # Legge la durata del video da overlay.mp4
                    overlay_path = os.path.join(subfolder, "overlay.mp4")
                    video_duration = get_video_duration(overlay_path)
                    data["video_duration_seconds"] = video_duration
                    
                    # Calcola il tempo di conversione medio per minuto di video
                    if video_duration > 0:
                        data["conversion_time_per_minute"] = data["total_processing_time"] * 60 / video_duration
                    else:
                        data["conversion_time_per_minute"] = 0
                        
                    # Calcola dimensioni originali (overlay.mp4) e compresso (compressed.mp4)
                    original_size = get_file_size(overlay_path)
                    compressed_path = os.path.join(subfolder, "compressed.mp4")
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