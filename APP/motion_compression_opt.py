import cv2
import numpy as np
import logging
import os
import time
from multiprocessing import Pool, cpu_count

def setup_logging(output_dir):
    """
    Sets up logging so that logs are both printed to the console
    and saved to a file named 'processing.log' in the specified directory.
    """
    log_file = os.path.join(output_dir, "processing.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
    )
    logging.info(f"Logging configured. Log file saved in: {log_file}")

def generate_bounding_box(mask):
    """
    Finds the contours of the white areas in the mask (representing motion).
    For each contour, it draws a filled rectangle (bounding box) on a blank mask of the same size.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_rect = np.zeros_like(mask)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(mask_rect, (x, y), (x + w, y + h), 255, -1)
    return mask_rect

def temporal_smoothing_flow(video_path, output_dir, flow_threshold=0.5, alpha_fraction=0.2, 
                            window_size=10, morph_kernel=2, save_name="overlay.mp4", 
                            mask_save_name="mask.mp4",
                            progress_callback_motion=None):
    """
    Reads a video, computes optical flow between consecutive frames, 
    and uses a temporal smoothing approach to detect motion. A bounding box mask
    is generated for areas exceeding the flow threshold, and this mask is smoothed 
    over a certain number of frames. 'overlay.mp4' stores the original frames, 
    and 'mask.mp4' stores the bounding box mask frames.

    If progress_callback_motion is provided, it will be called with a single float argument 
    in [0,1] representing the fraction of frames processed.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory where output videos will be saved.
        flow_threshold (float): Threshold for motion magnitude.
        alpha_fraction (float): Fraction of frames in the queue that must have motion 
                                for a pixel to be considered moving.
        window_size (int): Number of frames used for temporal smoothing.
        morph_kernel (int): Size of the morphological kernel for opening/closing operations.
        save_name (str): Name of the file where original frames (overlay) will be saved.
        mask_save_name (str): Name of the file where the bounding box mask will be saved.
        progress_callback_motion (callable): Optional function for updating progress.
    """
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Se non possiamo determinare i frame totali, assumiamo 1 per evitare divisioni per zero
        total_frames = 1

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

        # Aggiorniamo la progress bar solo ogni 10 frame, per non aggiungere overhead
        if progress_callback_motion and frame_count % 10 == 0:
            progress_fraction = min(frame_count / total_frames, 1.0)
            progress_callback_motion(progress_fraction)

    cap.release()
    out_overlay.release()
    out_mask.release()

    total_time = time.time() - start_time
    logging.info(f"Motion detection completed in {total_time:.2f} seconds. Frames processed: {frame_count}")

def process_frame(args):
    """
    Takes a frame from the original video and its corresponding mask frame.
    If the mask indicates no motion (mask area = 0), it applies a simplified 
    block-based DCT compression to that region to aggressively reduce detail. 
    It also sets the chrominance channels (Cr, Cb) to a neutral value (128) 
    in motionless blocks.
    
    Args:
        args (tuple): Contains (frame_in, frame_mask, QTY_aggressive).
    Returns:
        The processed BGR frame after applying block-level DCT compression 
        in static regions.
    """
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

def compress_with_motion(input_video, mask_video, output_dir, progress_callback_compression=None):
    """
    Reads the 'overlay' video (original frames) and the 'mask' video 
    (bounding box mask). Processes frames in batches, applying a more 
    aggressive block-based DCT compression to regions without motion.

    If progress_callback_compression is provided, it will be called with
    a float argument in [0,1] representing the fraction of frames processed.
    """
    start_time = time.time()

    cap_input = cv2.VideoCapture(input_video)
    cap_mask = cv2.VideoCapture(mask_video)

    total_frames_input = int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames_input <= 0:
        total_frames_input = 1

    fps = cap_input.get(cv2.CAP_PROP_FPS)
    width = int(cap_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = os.path.join(output_dir, "compressed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    QTY_aggressive = np.full((8, 8), 100, dtype=np.float32)
    batch_size = 400 #increase or decrease this parameter according to hw capabilities
    frames = []
    processed_frames_count = 0

    while True:
        ret_in, frame_in = cap_input.read()
        ret_mask, frame_mask = cap_mask.read()
        if not (ret_in and ret_mask):
            break
        frames.append((frame_in, frame_mask, QTY_aggressive))
        processed_frames_count += 1

        if len(frames) == batch_size:
            process_and_write_batch(frames, out)
            frames.clear()

        # Aggiorniamo ogni 10 frame
        if progress_callback_compression and processed_frames_count % 10 == 0:
            fraction = min(processed_frames_count / total_frames_input, 1.0)
            progress_callback_compression(fraction)

    if frames:
        process_and_write_batch(frames, out)

    cap_input.release()
    cap_mask.release()
    out.release()

    total_time = time.time() - start_time
    logging.info(f"Compression completed in {total_time:.2f} seconds.")

def process_and_write_batch(frames, out):
    """
    Uses a multiprocessing Pool to process each frame in parallel 
    and then writes the processed frames to the output video.
    """
    pool = Pool(processes=cpu_count())
    results = pool.map(process_frame, frames)
    pool.close()
    pool.join()

    for frame in results:
        out.write(frame)

def process_single_video(video_path, output_dir,
                         progress_callback_motion=None,
                         progress_callback_compression=None):
    """
    Combines the motion detection (with bounding boxes) and the subsequent 
    compression steps for a single video. Creates a specific output directory 
    for the video, sets up logging, performs motion detection, and then applies compression.

    If progress_callback_motion or progress_callback_compression are provided,
    they will be called with a float in [0,1] to indicate progress in each phase.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    setup_logging(video_output_dir)
    logging.info(f"=== Processing for video '{video_name}' started ===")

    # STEP 1: Motion Detection
    logging.info("Step 1/2: Starting motion detection...")
    start_md = time.time()
    temporal_smoothing_flow(
        video_path, 
        video_output_dir,
        progress_callback_motion=progress_callback_motion
    )
    md_time = time.time() - start_md
    logging.info(f"Step 1/2: Motion detection completed (elapsed: {md_time:.2f} s).")

    # STEP 2: Compression
    logging.info("Step 2/2: Starting compression...")
    start_comp = time.time()
    compress_with_motion(
        os.path.join(video_output_dir, "overlay.mp4"),
        os.path.join(video_output_dir, "mask.mp4"),
        video_output_dir,
        progress_callback_compression=progress_callback_compression
    )
    comp_time = time.time() - start_comp
    logging.info(f"Step 2/2: Compression completed (elapsed: {comp_time:.2f} s).")

    logging.info(f"=== Processing of '{video_name}' completed successfully. ===")

if __name__ == "__main__":
    process_single_video("./Dataset/input/video.mp4", "./Dataset/output/")
