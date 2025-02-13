import cv2
import numpy as np
import os
import logging
import time

def setup_logging(output_dir):
    """
    Configures logging: creates the output folder and sets up the processing.log file.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "processing.log")
    # If logging has already been configured elsewhere, this command might have no effect;
    # in any case, we set up a file handler and a stream handler.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])
    logging.info(f"Logging configured. Log file saved in: {log_file}")

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
    Performs block matching to detect motion and compress static areas.
    
    - video_path: input video path.
    - output_dir: base folder to save results.
    - progress_callback: (optional) function to report progress; called every 50 frames.
    """
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Unable to open the video.")
        return

    # Create a dedicated folder using the video name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # Configure logging in this subfolder
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
        logging.error("Unable to read the first frame of the video.")
        cap.release()
        return

    # Pre-process the first frame
    prev_frame = cv2.resize(prev_frame, (scaled_width, scaled_height))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # Use a more pronounced blur for the first frame (as in the original code)
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

            # Calculate the difference between the current frame and the previous one
            diff = cv2.absdiff(prev_gray, curr_gray)
            _, motion_mask = cv2.threshold(diff, motion_threshold, 255, cv2.THRESH_BINARY)

            # Find contours to eliminate small noise
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_mask = np.zeros_like(motion_mask)
            for contour in contours:
                if cv2.contourArea(contour) > min_area:
                    cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

            dilated_mask = cv2.dilate(filtered_mask, kernel, iterations=1)
            accumulated_mask = cv2.addWeighted(accumulated_mask, release_factor, dilated_mask, 1 - release_factor, 0)

            # Create an overlay that highlights the areas in motion
            mask_overlay = curr_frame.copy()
            mask_overlay[accumulated_mask > 127] = [0, 0, 255]
            mask_out.write(mask_overlay)

            # DCT compression in static areas (where no motion is detected)
            frame_ycrcb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2YCrCb)
            channels = cv2.split(frame_ycrcb)
            for y in range(0, scaled_height, block_size):
                for x in range(0, scaled_width, block_size):
                    # If the average of the mask block indicates no motion
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
        logging.error("Error during processing: " + str(e), exc_info=True)
    finally:
        cap.release()
        mask_out.release()
        final_out.release()

    total_time = time.time() - start_time
    avg_time_per_frame = (sum(frame_processing_times) / len(frame_processing_times)
                          if frame_processing_times else 0)

    # Writes the performance log file with the same structure as motion_compression_opt.py
    with open(time_log_path, "w") as f:
        f.write("Frame Differencing:\n")
        f.write(f"  Frames processed: {frame_count}\n")
        f.write(f"  Total time: {total_time:.2f} seconds\n")
        f.write(f"  Average time per frame: {avg_time_per_frame:.4f} seconds\n\n")
        f.write(f"Total video processing time: {total_time:.2f} seconds\n")

    logging.info(f"Execution statistics saved in: {time_log_path}")

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
    High-level function to process a video with the block matching method
    for frame differencing. Creates a subfolder (using the video name), configures logging,
    and calls the actual processing.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    setup_logging(video_output_dir)
    logging.info(f"=== Start processing (Frame Differencing) for '{video_name}' ===")

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

    logging.info(f"=== Processing successfully completed for '{video_name}'. ===")

if __name__ == "__main__":
    # Example standalone execution (modify paths as needed)
    process_single_video_fd(
        video_path="./Vandalism015_x264.mp4",
        output_dir="./output",
        block_size=8,
        kernel_size=10,
        release_factor=0.3,
        quantization_level=100,
        scale_factor=0.5
    )