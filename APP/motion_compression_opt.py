import cv2
import numpy as np
import logging
import os
import time
from multiprocessing import Pool, cpu_count

def setup_logging(output_dir):
    log_file = os.path.join(output_dir, "processing.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
    )
    logging.info(f"Logging configurato. File di log salvato in: {log_file}")

# Funzione per generare bounding box attorno agli oggetti in movimento
def generate_bounding_box(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_rect = np.zeros_like(mask)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(mask_rect, (x, y), (x + w, y + h), 255, -1)
    return mask_rect

# Motion Detection con bounding box rettangolari
def temporal_smoothing_flow(video_path, output_dir, flow_threshold=0.5, alpha_fraction=0.2, 
                            window_size=10, morph_kernel=2, save_name="overlay.mp4", 
                            mask_save_name="mask.mp4"):

    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
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
        logging.error("Errore: impossibile leggere il primo frame.")
        return

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    frame_count = 0
    mask_queue = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, pyr_scale=0.5, levels=1, winsize=5,
            iterations=1, poly_n=5, poly_sigma=1.1, flags=0
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
        mask_current = (mag > flow_threshold).astype(np.uint8) * 255
        mask_queue.append(mask_current)

        if len(mask_queue) > window_size:
            mask_queue.pop(0)

        cumulative_mask = np.sum(mask_queue, axis=0)
        mask_smoothed = (cumulative_mask >= (alpha_fraction * len(mask_queue) * 255)).astype(np.uint8) * 255
        mask_smoothed = cv2.morphologyEx(mask_smoothed, cv2.MORPH_CLOSE, kernel)
        mask_smoothed = cv2.morphologyEx(mask_smoothed, cv2.MORPH_OPEN, kernel)

        mask_rect = generate_bounding_box(mask_smoothed)
        out_overlay.write(frame)
        out_mask.write(mask_rect)
        prev_gray = gray.copy()
        frame_count += 1

    cap.release()
    out_overlay.release()
    out_mask.release()
    logging.info(f"Motion detection completata in {time.time() - start_time:.2f} secondi. Frames: {frame_count}")

# Funzione per elaborare un frame
def process_frame(args):
    frame_in, frame_mask, QTY_aggressive = args

    if len(frame_mask.shape) == 3:
        frame_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)

    frame_ycrcb = cv2.cvtColor(frame_in, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(frame_ycrcb)

    for i in range(0, frame_mask.shape[0], 8):
        for j in range(0, frame_mask.shape[1], 8):
            if frame_mask[i:i + 8, j:j + 8].mean() == 0:
                for channel in [y_channel, cr_channel, cb_channel]:
                    block = channel[i:i + 8, j:j + 8]
                    if block.shape == (8, 8):
                        dct_block = cv2.dct(block.astype(np.float32) - 128)
                        quantized_block = np.round(dct_block / QTY_aggressive) * QTY_aggressive
                        idct_block = cv2.idct(quantized_block) + 128
                        channel[i:i + 8, j:j + 8] = np.clip(idct_block, 0, 255)
                cr_channel[i:i + 8, j:j + 8] = 128
                cb_channel[i:i + 8, j:j + 8] = 128

    frame_compressed = cv2.merge([y_channel, cr_channel, cb_channel])
    return cv2.cvtColor(frame_compressed, cv2.COLOR_YCrCb2BGR)

# Compressione con gestione a batch
def compress_with_motion(input_video, mask_video, output_dir):
    cap_input = cv2.VideoCapture(input_video)
    cap_mask = cv2.VideoCapture(mask_video)

    fps = cap_input.get(cv2.CAP_PROP_FPS)
    width = int(cap_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = os.path.join(output_dir, "compressed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    QTY_aggressive = np.full((8, 8), 100, dtype=np.float32)
    batch_size = 200
    frames = []

    while True:
        ret_in, frame_in = cap_input.read()
        ret_mask, frame_mask = cap_mask.read()
        if not (ret_in and ret_mask):
            break
        frames.append((frame_in, frame_mask, QTY_aggressive))

        if len(frames) == batch_size:
            process_and_write_batch(frames, out)
            frames.clear()

    if frames:
        process_and_write_batch(frames, out)

    cap_input.release()
    cap_mask.release()
    out.release()

def process_and_write_batch(frames, out):
    pool = Pool(processes=cpu_count())
    results = pool.map(process_frame, frames)
    pool.close()
    pool.join()

    for frame in results:
        out.write(frame)

def process_single_video(video_path, output_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    setup_logging(video_output_dir)
    logging.info(f"=== Inizio elaborazione del video '{video_name}' ===")

    temporal_smoothing_flow(video_path, video_output_dir)
    compress_with_motion(
        os.path.join(video_output_dir, "overlay.mp4"),
        os.path.join(video_output_dir, "mask.mp4"),
        video_output_dir
    )

    logging.info(f"=== Elaborazione '{video_name}' completata. ===")

if __name__ == "__main__":
    process_single_video("./Dataset/input/video.mp4", "./Dataset/output/")
