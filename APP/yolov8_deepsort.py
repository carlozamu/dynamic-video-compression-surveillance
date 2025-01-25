import cv2
import os
import logging
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def setup_logging(output_dir):
    """Setup logging for the application."""
    log_file = os.path.join(output_dir, "processing.log")
    file_handler = logging.FileHandler(log_file, mode='w', delay=False)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    file_handler.setLevel(logging.DEBUG)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    logging.info(f"Logging configured. Log file saved in: {log_file}")


def process_video_with_yolo_and_deepsort(video_path, output_dir, yolo_model='yolov8s.pt', display=False):
    """Process video using YOLO for detection and DeepSORT for tracking."""

    # Verify input video exists
    if not os.path.exists(video_path):
        logging.error(f"Input video file not found: {video_path}")
        return

    # Initialize YOLO and DeepSORT
    try:
        model = YOLO(yolo_model)
        tracker = DeepSort(max_age=30, nn_budget=70, nms_max_overlap=1.0)
        logging.info("YOLO and DeepSORT initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing YOLO or DeepSORT: {e}", exc_info=True)
        return

    # Setup video reading and writing
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = os.path.join(output_dir, "tracked_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.info("End of video reached or unable to read frame.")
                break

            frame_count += 1

            # YOLO detection
            results = model.predict(frame, conf=0.4)  # Set a lower confidence threshold if needed
            detections = []

            # Parse YOLO results
            for box in results[0].boxes.data.tolist():
                if len(box) >= 6:  # Ensure the box has all necessary fields
                    try:
                        x_min, y_min, x_max, y_max, conf, cls = box[:6]
                        detection = (
                            float(x_min),
                            float(y_min),
                            float(x_max),
                            float(y_max),
                            float(conf),
                            int(cls)
                        )
                        detections.append(detection)
                    except Exception as e:
                        logging.error(f"Error processing detection {box}: {e}")
                else:
                    logging.warning(f"Incomplete detection: {box}")

            logging.debug(f"Formatted detections passed to DeepSORT: {detections}")

            # DeepSORT tracking
            if detections:
                try:
                    tracks = tracker.update_tracks(detections, frame=frame)
                except Exception as e:
                    logging.error(f"Error during DeepSORT tracking: {e}")
                    tracks = []
            else:
                logging.warning("No detections to pass to DeepSORT.")
                tracks = []

            # Draw tracking results
            for track in tracks:
                if not track.is_confirmed():
                    logging.debug(f"Frame {frame_count}: Track {track.track_id} not confirmed.")
                    continue
                track_id = track.track_id
                l, t, r, b = map(int, track.to_tlbr())
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save and optionally display the frame
            out.write(frame)
            if display:
                cv2.imshow("Tracked Video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Processing interrupted by user.")
                    break

            if frame_count % 50 == 0:
                logging.info(f"Processed {frame_count} frames so far...")

    except Exception as e:
        logging.error(f"Unexpected error during video processing: {e}", exc_info=True)

    finally:
        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

    logging.info(f"Processed {frame_count} frames. Output saved to {output_video_path}.")


def main():
    """Main entry point for the script."""
    video_path = "C:/Users/gianl/Documents/AIS/Signals/PROJECT/dynamic-video-compression-surveillance/Dataset/input/Burglary009_x264.mp4"  # Replace with your input video path
    output_dir = "C:/Users/gianl/Documents/AIS/Signals/PROJECT/dynamic-video-compression-surveillance/Dataset/output/yolo_deepsort"  # Replace with your output directory
    os.makedirs(output_dir, exist_ok=True)

    setup_logging(output_dir)
    logging.info("Starting YOLO + DeepSORT video processing.")

    try:
        process_video_with_yolo_and_deepsort(
            video_path=video_path,
            output_dir=output_dir,
            yolo_model='yolov8s.pt',
            display=False  # Set to True to display the video during processing
        )
    except Exception as e:
        logging.error(f"Error in main processing: {e}", exc_info=True)

    logging.info("Processing completed.")


if __name__ == "__main__":
    main()
