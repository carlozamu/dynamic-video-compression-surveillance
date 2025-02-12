import cv2
import numpy as np
import logging
import os
import time
from collections import deque

def setup_logging(output_dir):
    """
    Configura il logging aggiungendo un file handler, 
    senza rimuovere eventuali handler già impostati (es. QtLogHandler).
    """
    log_file = os.path.join(output_dir, "processing.log")
    logger = logging.getLogger()

    # Verifica se esiste già un handler per il file, altrimenti aggiungilo.
    file_handler_exists = any(
        isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(log_file)
        for handler in logger.handlers
    )
    if not file_handler_exists:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    logger.info(f"Logging configured. Log file saved in: {log_file}")

def temporal_smoothing_flow(video_path, output_dir, flow_threshold=0.5, alpha_fraction=0.2, 
                              window_size=30, morph_kernel=2, save_name="overlay.mp4", 
                              mask_save_name="mask.mp4"):
    """
    Esegue il rilevamento del movimento con optical flow e smussamento temporale.
    Salva:
      - overlay.mp4: il video originale
      - mask.mp4: la maschera ottenuta dopo operazioni morfologiche e rettangolarizzazione
    """
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Unable to open video file: {video_path}")
        return 0, 0, 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    overlay_path = os.path.join(output_dir, save_name)
    mask_path = os.path.join(output_dir, mask_save_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_overlay = cv2.VideoWriter(overlay_path, fourcc, fps, (width, height))
    out_mask = cv2.VideoWriter(mask_path, fourcc, fps, (width, height), isColor=False)
    
    ret, first_frame = cap.read()
    if not ret:
        logging.error("Error: Unable to read the first frame.")
        cap.release()
        return 0, 0, 0
    
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    mask_queue = deque(maxlen=window_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.3,   # pyr_scale
            2,     # levels
            9,     # winsize
            2,     # iterations
            5,     # poly_n
            1.1,   # poly_sigma
            0      # flags
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask_current = (mag > flow_threshold).astype(np.uint8) * 255
        mask_queue.append(mask_current)
        cumulative_mask = np.sum(np.array(mask_queue), axis=0)
        mask_smoothed = (cumulative_mask >= (alpha_fraction * len(mask_queue) * 255)).astype(np.uint8) * 255
        
        # Operazioni morfologiche
        mask_smoothed = cv2.morphologyEx(mask_smoothed, cv2.MORPH_CLOSE, kernel)
        mask_smoothed = cv2.morphologyEx(mask_smoothed, cv2.MORPH_OPEN, kernel)
        
        # Rettangolarizza le aree di movimento
        contours, _ = cv2.findContours(mask_smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_rect = np.zeros((height, width), dtype=np.uint8)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(mask_rect, (x, y), (x + w, y + h), 255, -1)
        
        out_overlay.write(frame)
        out_mask.write(mask_rect)
        prev_gray = gray.copy()
    
    cap.release()
    out_overlay.release()
    out_mask.release()
    total_time = time.time() - start_time
    avg_time = total_time / frame_count if frame_count > 0 else 0
    logging.info(f"Temporal smoothing flow completed for '{os.path.basename(video_path)}' in {total_time:.2f} seconds. Frames processed: {frame_count}")
    return frame_count, total_time, avg_time

def compress_with_motion(input_video, mask_video, output_dir):
    """
    Applica la compressione basata sul movimento:
      - Nelle aree statiche (maschera == 0) viene applicata una compressione DCT aggressiva
        seguita dalla conversione in scala di grigi.
    Salva il video compresso in "compressed.mp4".
    """
    start_time = time.time()
    logging.info(f"Starting motion-based compression for: {os.path.basename(input_video)}")
    
    cap_input = cv2.VideoCapture(input_video)
    cap_mask = cv2.VideoCapture(mask_video)
    if not cap_input.isOpened():
        logging.error(f"Error: Unable to open input video: {input_video}")
        return 0, 0, 0
    if not cap_mask.isOpened():
        logging.error(f"Error: Unable to open mask video: {mask_video}")
        return 0, 0, 0
        
    fps = cap_input.get(cv2.CAP_PROP_FPS)
    width = int(cap_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_video = os.path.join(output_dir, "compressed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    QTY_aggressive = np.full((8, 8), 100, dtype=np.float32)
    frame_count = 0
    
    while True:
        ret_in, frame_in = cap_input.read()
        ret_mask, frame_mask = cap_mask.read()
        if not (ret_in and ret_mask):
            break
        frame_count += 1
        
        if len(frame_mask.shape) == 3:
            frame_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
        
        # Conversione del frame in YCrCb e separazione dei canali
        frame_ycrcb = cv2.cvtColor(frame_in, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(frame_ycrcb)
        
        # Per ogni blocco 8x8 nelle aree statiche (maschera media == 0) applica DCT + quantizzazione
        for i in range(0, frame_mask.shape[0], 8):
            for j in range(0, frame_mask.shape[1], 8):
                block_mask = frame_mask[i:i+8, j:j+8]
                if block_mask.size == 0 or block_mask.shape[0] < 8 or block_mask.shape[1] < 8:
                    continue
                if block_mask.mean() == 0:
                    for c in range(3):
                        block = channels[c][i:i+8, j:j+8]
                        if block.shape == (8, 8):
                            dct_block = cv2.dct(block.astype(np.float32) - 128)
                            quantized_block = np.round(dct_block / QTY_aggressive) * QTY_aggressive
                            idct_block = cv2.idct(quantized_block) + 128
                            channels[c][i:i+8, j:j+8] = np.clip(idct_block, 0, 255)
        
        frame_processed = cv2.merge(channels)
        frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_YCrCb2BGR)
        
        # Converte in scala di grigi i blocchi statici
        for i in range(0, frame_mask.shape[0], 8):
            for j in range(0, frame_mask.shape[1], 8):
                block_mask = frame_mask[i:i+8, j:j+8]
                if block_mask.size == 0 or block_mask.shape[0] < 8 or block_mask.shape[1] < 8:
                    continue
                if block_mask.mean() == 0:
                    roi = frame_processed[i:i+8, j:j+8]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    gray_roi_bgr = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)
                    frame_processed[i:i+8, j:j+8] = gray_roi_bgr
        
        out.write(frame_processed)
    
    cap_input.release()
    cap_mask.release()
    out.release()
    total_time = time.time() - start_time
    avg_time = total_time / frame_count if frame_count > 0 else 0
    logging.info(f"Motion-based compression completed for '{os.path.basename(input_video)}' in {total_time:.2f} seconds. Frames processed: {frame_count}")
    return frame_count, total_time, avg_time

def process_single_video_of(video_path, output_dir):
    """
    Combina i passaggi di motion detection e compressione per un singolo video.
    - Crea una sottocartella dedicata (usando il nome del video)
    - Imposta il logging in quella cartella
    - Esegue motion detection e compressione
    - Registra i tempi di esecuzione in 'execution_times.txt'
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    setup_logging(video_output_dir)
    logging.info(f"=== Processing for video '{video_name}' started ===")
    
    # STEP 1: Motion Detection
    logging.info("Step 1/2: Starting motion detection...")
    md_frames, md_time, md_avg = temporal_smoothing_flow(
        video_path, 
        video_output_dir,
        flow_threshold=0.5,
        alpha_fraction=0.2,
        window_size=30,
        morph_kernel=2,
        save_name="overlay.mp4",
        mask_save_name="mask.mp4"
    )
    logging.info(f"Step 1/2: Motion detection completed (elapsed: {md_time:.2f} s, avg per frame: {md_avg:.4f} s).")
    
    # STEP 2: Compression
    logging.info("Step 2/2: Starting compression...")
    cp_frames, cp_time, cp_avg = compress_with_motion(
        os.path.join(video_output_dir, "overlay.mp4"),
        os.path.join(video_output_dir, "mask.mp4"),
        video_output_dir
    )
    logging.info(f"Step 2/2: Compression completed (elapsed: {cp_time:.2f} s, avg per frame: {cp_avg:.4f} s).")
    
    total_processing_time = md_time + cp_time
    execution_times_path = os.path.join(video_output_dir, "execution_times.txt")
    with open(execution_times_path, "w") as f:
        f.write("Motion Detection:\n")
        f.write(f"  Frames processed: {md_frames}\n")
        f.write(f"  Total time: {md_time:.2f} seconds\n")
        f.write(f"  Average time per frame: {md_avg:.4f} seconds\n\n")
        f.write("Compression:\n")
        f.write(f"  Frames processed: {cp_frames}\n")
        f.write(f"  Total time: {cp_time:.2f} seconds\n")
        f.write(f"  Average time per frame: {cp_avg:.4f} seconds\n\n")
        f.write(f"Total video processing time: {total_processing_time:.2f} seconds\n")
    
    logging.info(f"Execution times logged in: {execution_times_path}")
    logging.info(f"=== Processing of '{video_name}' completed successfully. ===")

if __name__ == "__main__":
    process_single_video_of("./Dataset/input/video.mp4", "./Dataset/output/")