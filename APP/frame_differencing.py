import cv2
import numpy as np
import os
import logging
import time

def setup_logging(output_dir):
    """
    Configura il logging: crea la cartella di output e imposta il file processing.log.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "processing.log")
    # Se il logging è già stato configurato altrove, questo comando potrebbe non avere effetto;
    # in ogni caso, impostiamo un handler file e uno stream.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])
    logging.info(f"Logging configurato. File di log salvato in: {log_file}")

def filter_and_dilate_movements(video_path, output_dir, 
                                block_size=4,
                                search_area=16,
                                motion_threshold=0.5,
                                min_area=500,
                                kernel_size=7,
                                release_factor=0.5,
                                quantization_level=100,
                                scale_factor=1.0,
                                progress_callback=None):
    """
    Esegue il block matching per rilevare il movimento e comprimere le zone statiche.
    
    - video_path: percorso del video di input.
    - output_dir: cartella base per salvare i risultati.
    - progress_callback: (opzionale) funzione per comunicare l'avanzamento; viene chiamata ogni 50 frame.
    """
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Impossibile aprire il video.")
        return

    # Crea una cartella dedicata usando il nome del video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # Configura il logging in questa sottocartella
    setup_logging(video_output_dir)

    mask_output_path = os.path.join(video_output_dir, "dilated_motion_mask_video.mp4")
    final_output_path = os.path.join(video_output_dir, "compressed_final_video.mp4")
    time_log_path = os.path.join(video_output_dir, "execution_times.txt")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scaled_width = int(width * scale_factor)
    scaled_height = int(height * scale_factor)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    mask_out = cv2.VideoWriter(mask_output_path, fourcc, fps, (scaled_width, scaled_height))
    final_out = cv2.VideoWriter(final_output_path, fourcc, fps, (scaled_width, scaled_height))

    ret, prev_frame = cap.read()
    if not ret:
        logging.error("Impossibile leggere il primo frame del video.")
        cap.release()
        return

    # Pre-processamento del primo frame
    prev_frame = cv2.resize(prev_frame, (scaled_width, scaled_height))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # Uso di un blur più marcato per il primo frame (come nel codice originale)
    prev_gray = cv2.GaussianBlur(prev_gray, (25, 25), 30)

    frame_count = 0
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    accumulated_mask = np.zeros((scaled_height, scaled_width), dtype=np.uint8)
    frame_processing_times = []

    try:
        while True:
            frame_start = time.time()
            ret, curr_frame = cap.read()
            if not ret:
                break

            curr_frame = cv2.resize(curr_frame, (scaled_width, scaled_height))
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

            # Calcola la differenza tra il frame corrente e quello precedente
            diff = cv2.absdiff(prev_gray, curr_gray)
            _, motion_mask = cv2.threshold(diff, motion_threshold, 255, cv2.THRESH_BINARY)

            # Trova i contorni per eliminare piccoli rumori
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_mask = np.zeros_like(motion_mask)
            for contour in contours:
                if cv2.contourArea(contour) > min_area:
                    cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

            dilated_mask = cv2.dilate(filtered_mask, kernel, iterations=1)
            accumulated_mask = cv2.addWeighted(accumulated_mask, release_factor, dilated_mask, 1 - release_factor, 0)

            # Crea un overlay che evidenzia le aree in movimento
            mask_overlay = curr_frame.copy()
            mask_overlay[accumulated_mask > 127] = [0, 0, 255]
            mask_out.write(mask_overlay)

            # Compressione DCT nelle zone statiche (dove non è stato rilevato movimento)
            frame_ycrcb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2YCrCb)
            channels = cv2.split(frame_ycrcb)
            for y in range(0, scaled_height, block_size):
                for x in range(0, scaled_width, block_size):
                    # Se la media del blocco della maschera indica assenza di movimento
                    if accumulated_mask[y:y+block_size, x:x+block_size].mean() == 0:
                        block = channels[0][y:y+block_size, x:x+block_size]
                        dct_block = cv2.dct(block.astype(np.float32) - 128)
                        quantized_block = np.round(dct_block / quantization_level) * quantization_level
                        idct_block = cv2.idct(quantized_block) + 128
                        channels[0][y:y+block_size, x:x+block_size] = np.clip(idct_block, 0, 255)
                        channels[1][y:y+block_size, x:x+block_size] = 128
                        channels[2][y:y+block_size, x:x+block_size] = 128

            compressed_frame = cv2.merge(channels)
            compressed_frame = cv2.cvtColor(compressed_frame, cv2.COLOR_YCrCb2BGR)
            final_out.write(compressed_frame)

            prev_gray = curr_gray.copy()
            frame_count += 1
            frame_processing_times.append(time.time() - frame_start)

            if progress_callback is not None and frame_count % 50 == 0:
                progress_callback(frame_count)

    except Exception as e:
        logging.error("Errore durante l'elaborazione: " + str(e), exc_info=True)
    finally:
        cap.release()
        mask_out.release()
        final_out.release()

    total_time = time.time() - start_time
    avg_time_per_frame = (sum(frame_processing_times) / len(frame_processing_times)
                          if frame_processing_times else 0)

    # Scrive il file di log delle performance con la stessa struttura di motion_compression_opt.py
    with open(time_log_path, "w") as f:
        f.write("Frame Differencing:\n")
        f.write(f"  Frames processed: {frame_count}\n")
        f.write(f"  Total time: {total_time:.2f} seconds\n")
        f.write(f"  Average time per frame: {avg_time_per_frame:.4f} seconds\n\n")
        f.write(f"Total video processing time: {total_time:.2f} seconds\n")

    logging.info(f"Statistiche di esecuzione salvate in: {time_log_path}")

def process_single_video_fd(video_path,
                            output_dir,
                            block_size=4,
                            search_area=16,
                            motion_threshold=0.5,
                            min_area=500,
                            kernel_size=7,
                            release_factor=0.5,
                            quantization_level=100,
                            scale_factor=1.0,
                            progress_callback=None):
    """
    Funzione di alto livello per processare un video con il metodo del block matching
    per frame differencing. Crea una sottocartella (usando il nome del video), configura il logging
    e richiama l'elaborazione vera e propria.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    setup_logging(video_output_dir)
    logging.info(f"=== Inizio elaborazione (Frame Differencing) per '{video_name}' ===")

    filter_and_dilate_movements(video_path,
                                output_dir,
                                block_size=block_size,
                                search_area=search_area,
                                motion_threshold=motion_threshold,
                                min_area=min_area,
                                kernel_size=kernel_size,
                                release_factor=release_factor,
                                quantization_level=quantization_level,
                                scale_factor=scale_factor,
                                progress_callback=progress_callback)

    logging.info(f"=== Elaborazione completata con successo per '{video_name}'. ===")

if __name__ == "__main__":
    # Esempio di esecuzione standalone (modifica i percorsi in base alle tue esigenze)
    process_single_video_fd(
        video_path="./Vandalism015_x264.mp4",
        output_dir="./output",
        block_size=8,
        kernel_size=10,
        release_factor=0.3,
        quantization_level=100,
        scale_factor=0.5
    )