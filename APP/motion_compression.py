import cv2
import numpy as np
import logging
import os
import time
from collections import deque

def setup_logging(output_dir):
    log_file = os.path.join(output_dir, "processing.log")
    
    # Rimuovi tutti i vecchi handler, se esistono
    root_logger = logging.getLogger()
    while root_logger.hasHandlers():
        root_logger.removeHandler(root_logger.handlers[0])
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w', delay=False),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging configurato. File di log salvato in: {log_file}")

def temporal_smoothing_flow(
    video_path,
    output_dir,
    flow_threshold=0.5,
    alpha_fraction=0.2,
    window_size=30,
    morph_kernel=2,
    save_name="overlay.mp4",
    mask_save_name="mask.mp4"
):
    """
    Calcola il flusso ottico Farneback sull'intero video e salva:
      - overlay.mp4  (video originale)
      - mask.mp4     (maschera di movimento rettangolarizzata)
    """
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Impossibile aprire il video in ingresso: {video_path}")
        return

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
        logging.error("Impossibile leggere il primo frame del video.")
        cap.release()
        return

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    from collections import deque
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
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
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

    end_time = time.time()
    logging.info(
        f"Temporal smoothing flow completed for '{os.path.basename(video_path)}' "
        f"in {end_time - start_time:.2f} seconds. Frame elaborati: {frame_count}"
    )

def compress_with_motion(input_video, mask_video, output_dir):
    """
    Compressione basata sul movimento:
    - Aree con movimento: conserva i frame a colori
    - Aree senza movimento: DCT aggressiva + conversione in B/N
    """
    start_time = time.time()
    logging.info(f"Avvio della motion-based compression per: {os.path.basename(input_video)}")

    cap_input = cv2.VideoCapture(input_video)
    cap_mask = cv2.VideoCapture(mask_video)

    if not cap_input.isOpened():
        logging.error(f"Impossibile aprire il video originale: {input_video}")
        return
    if not cap_mask.isOpened():
        logging.error(f"Impossibile aprire il video maschera: {mask_video}")
        return

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

        # Converti frame in YCrCb
        frame_ycrcb = cv2.cvtColor(frame_in, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(frame_ycrcb)

        # Applica DCT + quantizzazione aggressiva nelle zone statiche
        for i in range(0, frame_mask.shape[0], 8):
            for j in range(0, frame_mask.shape[1], 8):
                if frame_mask[i:i+8, j:j+8].mean() == 0:
                    for c in range(3):
                        block = channels[c][i:i+8, j:j+8]
                        if block.shape == (8, 8):
                            dct_block = cv2.dct(block.astype(np.float32) - 128)
                            quantized_block = np.round(dct_block / QTY_aggressive) * QTY_aggressive
                            idct_block = cv2.idct(quantized_block) + 128
                            channels[c][i:i+8, j:j+8] = np.clip(idct_block, 0, 255)

        frame_processed = cv2.merge(channels)
        frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_YCrCb2BGR)

        # Converte in B/N i blocchi statici
        for i in range(0, frame_mask.shape[0], 8):
            for j in range(0, frame_mask.shape[1], 8):
                if frame_mask[i:i+8, j:j+8].mean() == 0:
                    roi = frame_processed[i:i+8, j:j+8]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    gray_roi_bgr = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)
                    frame_processed[i:i+8, j:j+8] = gray_roi_bgr

        out.write(frame_processed)

    cap_input.release()
    cap_mask.release()
    out.release()

    end_time = time.time()
    logging.info(
        f"Motion-based compression completed for '{os.path.basename(input_video)}' "
        f"in {end_time - start_time:.2f} seconds. Frame elaborati: {frame_count}"
    )


def process_single_video(video_path, output_dir):
    """
    Elabora UN SOLO video, usando temporal_smoothing_flow e compress_with_motion.
    - Crea una sottocartella in output con lo stesso nome del file.
    - Imposta il logging in quella cartella.
    - Esegue il flusso ottico -> genera overlay e mask.
    - Esegue la compressione basata sul movimento -> genera compressed.mp4.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # 1. Setup logging per il singolo video
    setup_logging(video_output_dir)
    logging.info(f"=== Inizio elaborazione del video '{video_name}' ===")

    # 2. Esegui temporal_smoothing_flow
    temporal_smoothing_flow(
        video_path=video_path,
        output_dir=video_output_dir,
        flow_threshold=0.5,
        alpha_fraction=0.2,
        window_size=30,
        morph_kernel=2,
        save_name=f"{video_name}_overlay.mp4",
        mask_save_name=f"{video_name}_mask.mp4"
    )

    # 3. Esegui compress_with_motion
    compress_with_motion(
        input_video=os.path.join(video_output_dir, f"{video_name}_overlay.mp4"),
        mask_video=os.path.join(video_output_dir, f"{video_name}_mask.mp4"),
        output_dir=video_output_dir
    )

    logging.info(f"=== Elaborazione '{video_name}' completata. ===")


def main():
    """
    Esegue l'elaborazione per TUTTI i file .mp4 in ./Dataset/input,
    salvando i risultati in ./Dataset/output/<nome_video>.
    """
    start_time_global = time.time()
    input_dir = "./Dataset/input/"
    output_dir = "./Dataset/output/"

    if not os.path.exists(input_dir):
        print(f"Directory di input non trovata: {os.path.abspath(input_dir)}")
        return

    video_list = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")]
    if not video_list:
        print("Nessun file .mp4 trovato nella directory di input.")
        return

    print(f"Video trovati: {video_list}")

    # Esegui per ogni video
    for video in video_list:
        video_path = os.path.join(input_dir, video)
        process_single_video(video_path, output_dir)

    end_time_global = time.time()
    print(f"Elaborazione di tutti i video completata in {end_time_global - start_time_global:.2f} secondi.")

if __name__ == "__main__":
    main()