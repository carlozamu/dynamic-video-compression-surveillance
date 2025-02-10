import cv2
import numpy as np
import os
import time
import logging

def setup_logging(output_dir):
    """
    Imposta il logging sia su file che su console,
    salvando il log in 'processing.log' nella cartella specificata.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "processing.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
    )
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
                                progress_callback_motion=None,
                                progress_callback_compression=None):
    """
    Esegue motion detection e compressione DCT in un unico passaggio.
    Le callback di avanzamento vengono aggiornate ogni 50 frame e i messaggi di log
    vengono stampati ogni 100 frame per minimizzare l’overhead.
    """
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1  # fallback per evitare divisione per zero

    if not cap.isOpened():
        logging.error("Impossibile aprire il video.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    mask_output_path = os.path.join(output_dir, "dilated_motion_mask_video.mp4")
    final_output_path = os.path.join(output_dir, "compressed_final_video.mp4")
    time_log_path = os.path.join(output_dir, "execution_times.txt")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
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
    
    # Ridimensiona e applica un blur meno pesante (più veloce)
    prev_frame = cv2.resize(prev_frame, (scaled_width, scaled_height))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)  # Era (25,25) con sigma=30
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    accumulated_mask = np.zeros((scaled_height, scaled_width), dtype=np.uint8)
    
    frame_count = 0
    frame_processing_times = []

    while True:
        frame_start = time.time()
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        curr_frame = cv2.resize(curr_frame, (scaled_width, scaled_height))
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
        
        # Calcola la differenza e applica la soglia
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, motion_mask = cv2.threshold(diff, motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Trova e filtra i contorni
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(motion_mask)
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Dilata e aggiorna la maschera accumulata
        dilated_mask = cv2.dilate(filtered_mask, kernel, iterations=1)
        accumulated_mask = cv2.addWeighted(accumulated_mask, release_factor, dilated_mask, 1 - release_factor, 0)
        
        # Crea l'overlay e scrive il frame
        mask_overlay = curr_frame.copy()
        mask_overlay[accumulated_mask > 127] = [0, 0, 255]
        mask_out.write(mask_overlay)
        
        # Compressione DCT solo sulle zone statiche
        frame_ycrcb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(frame_ycrcb)
        for y in range(0, scaled_height, block_size):
            for x in range(0, scaled_width, block_size):
                if accumulated_mask[y:y + block_size, x:x + block_size].mean() == 0:
                    block = y_channel[y:y + block_size, x:x + block_size]
                    dct_block = cv2.dct(block.astype(np.float32) - 128)
                    quantized_block = np.round(dct_block / quantization_level) * quantization_level
                    idct_block = cv2.idct(quantized_block) + 128
                    y_channel[y:y + block_size, x:x + block_size] = np.clip(idct_block, 0, 255)
                    
                    cr_channel[y:y + block_size, x:x + block_size] = 128
                    cb_channel[y:y + block_size, x:x + block_size] = 128
        
        compressed_frame = cv2.merge([y_channel, cr_channel, cb_channel])
        compressed_frame = cv2.cvtColor(compressed_frame, cv2.COLOR_YCrCb2BGR)
        final_out.write(compressed_frame)
        
        prev_gray = curr_gray.copy()
        frame_count += 1
        frame_processing_times.append(time.time() - frame_start)
        
        # Aggiornamento delle progress bar ogni 50 frame (invece che ogni 10)
        if progress_callback_motion and frame_count % 50 == 0:
            fraction = min(frame_count / total_frames, 1.0)
            progress_callback_motion(fraction)
        if progress_callback_compression and frame_count % 50 == 0:
            fraction = min(frame_count / total_frames, 1.0)
            progress_callback_compression(fraction)
        
        # Log ogni 100 frame per ridurre l’overhead
        if frame_count % 100 == 0:
            logging.info(f"Frame elaborati: {frame_count}/{total_frames}")

    cap.release()
    mask_out.release()
    final_out.release()
    
    total_time = time.time() - start_time
    avg_time_per_frame = (sum(frame_processing_times) / len(frame_processing_times)) if frame_processing_times else 0
    
    with open(time_log_path, "w") as f:
        f.write(f"Tempo totale: {total_time:.2f} secondi\n")
        f.write(f"Tempo medio per frame: {avg_time_per_frame:.4f} secondi\n")
    
    logging.info(f"Elaborazione completata in {total_time:.2f} secondi.")
    logging.info(f"File salvati in: {output_dir}")
    logging.info(f"Statistiche di esecuzione in: {time_log_path}")

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
                            progress_callback_motion=None,
                            progress_callback_compression=None):
    """
    Funzione ad alto livello per processare un video con frame differencing.
    Crea una cartella dedicata (usando il nome del video), configura il logging,
    e richiama la funzione di elaborazione.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    setup_logging(video_output_dir)
    logging.info(f"=== Inizio elaborazione (metodo differenze) per '{video_name}' ===")

    filter_and_dilate_movements(video_path,
                                video_output_dir,
                                block_size=block_size,
                                search_area=search_area,
                                motion_threshold=motion_threshold,
                                min_area=min_area,
                                kernel_size=kernel_size,
                                release_factor=release_factor,
                                quantization_level=quantization_level,
                                scale_factor=scale_factor,
                                progress_callback_motion=progress_callback_motion,
                                progress_callback_compression=progress_callback_compression)

    logging.info(f"=== Elaborazione completata con successo per '{video_name}'. ===")

if __name__ == "__main__":
    process_single_video_fd(
        video_path="C:/Users/gianl/Documents/AIS/Signals/PROJECT/DATASET/INPUT/Vandalism015_x264.mp4",
        output_dir="C:/Users/gianl/Documents/AIS/Signals/PROJECT/DATASET/OUTPUT/BLOCK_MATCHING",
        block_size=8,
        kernel_size=10,
        release_factor=0.3,
        quantization_level=100,
        scale_factor=0.5
    )