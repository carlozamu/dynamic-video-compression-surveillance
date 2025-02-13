#!/usr/bin/env python
import os
import sys
import re
import csv
import matplotlib.pyplot as plt
import cv2  # Necessary to read video properties

def parse_execution_times(file_path):
    """
    Extracts performance data from the execution_times.txt file.
    
    For the Optical Flow format (execution_times.txt produced by motion_compression_opt.py)
      Motion Detection:
        Frames processed: <md_frames>
        Total time: <md_time> seconds
        Average time per frame: <md_avg> seconds

      Compression:
        Frames processed: <cp_frames>
        Total time: <cp_time> seconds
        Average time per frame: <cp_avg> seconds

      Total video processing time: <total_processing_time> seconds

    For the Frame Differencing format (execution_times.txt produced by frame_differencing.py)
      Frame Differencing:
        Frames processed: <frames>
        Total time: <time> seconds
        Average time per frame: <avg> seconds

      Total video processing time: <total_processing_time> seconds

    In the case of Frame Differencing, the values will be assigned to the md_* keys
    and the compression values (cp_*) will be set to 0.
    """
    data = {}
    try:
        with open(file_path, "r") as f:
            # Reads lines ignoring empty ones
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
            # Finds the Compression section:
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
            # Finds the total time line
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

            # Assigns Frame Differencing values to md_* keys
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
            raise ValueError("Unrecognized format of execution_times.txt")
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None
    return data

def get_video_duration(video_path):
    """Returns the duration of the video in seconds using cv2."""
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
    """Returns the file size in bytes."""
    try:
        return os.path.getsize(file_path)
    except Exception:
        return 0

def get_original_and_compressed_paths(subfolder):
    """
    Returns a tuple (original_path, compressed_path) based on the presence
    of files in the subfolders. If 'overlay.mp4' and 'compressed.mp4' exist, it uses them;
    otherwise, it tries with 'dilated_motion_mask_video.mp4' and 'compressed_final_video.mp4'.
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
        print("Usage: python performance_analysis.py <output_folder>")
        sys.exit(1)
        
    output_folder = sys.argv[1]
    if not os.path.isdir(output_folder):
        print(f"Invalid output folder: {output_folder}")
        sys.exit(1)
    
    # Create the 'performance' subfolder
    performance_folder = os.path.join(output_folder, "performance")
    os.makedirs(performance_folder, exist_ok=True)
    
    performance_data = []
    
    # Iterate through all subfolders in the output folder
    for item in os.listdir(output_folder):
        subfolder = os.path.join(output_folder, item)
        if os.path.isdir(subfolder):
            exec_file = os.path.join(subfolder, "execution_times.txt")
            if os.path.isfile(exec_file):
                data = parse_execution_times(exec_file)
                if data is not None:
                    data["video"] = item  # Name of the subfolder (video)
                    
                    # Get paths for the "original" and "compressed" videos
                    original_path, compressed_path = get_original_and_compressed_paths(subfolder)
                    if original_path is None or compressed_path is None:
                        print(f"Warning: video files not found in {subfolder}")
                        continue
                    
                    # Read the duration of the original video
                    video_duration = get_video_duration(original_path)
                    data["video_duration_seconds"] = video_duration
                    
                    # Calculate the average conversion time per minute of video
                    if video_duration > 0:
                        data["conversion_time_per_minute"] = data["total_processing_time"] * 60 / video_duration
                    else:
                        data["conversion_time_per_minute"] = 0
                        
                    # Calculate original and compressed sizes
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
        print("No performance data found.")
        sys.exit(1)
    
    # Define extended names for the CSV
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
    print(f"CSV saved in: {csv_file}")
    
    # Line chart: Total conversion time and conversion per minute for each video
    videos = [d["video"] for d in performance_data]
    total_times = [d["total_processing_time"] for d in performance_data]
    conv_time_per_min = [d["conversion_time_per_minute"] for d in performance_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(videos, total_times, marker='o', label="Total Conversion Time (s)")
    plt.plot(videos, conv_time_per_min, marker='o', label="Conversion Time per Minute (s/min)")
    plt.xlabel("Video")
    plt.ylabel("Time (s)")
    plt.title("Total Conversion Time and per Minute per Video")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    line_chart_path = os.path.join(performance_folder, "conversion_times_line_chart.png")
    plt.savefig(line_chart_path)
    plt.close()
    print(f"Line chart saved in: {line_chart_path}")
    
    # Bar chart: Reduction percentage for each video with overall average
    reduction_percentages = [d["reduction_percentage"] for d in performance_data]
    avg_reduction = sum(reduction_percentages) / len(reduction_percentages)
    
    plt.figure(figsize=(10, 6))
    plt.bar(videos, reduction_percentages, color="cornflowerblue", label="Reduction (%)")
    plt.axhline(y=avg_reduction, color="red", linestyle="--", label=f"Average Reduction ({avg_reduction:.2f}%)")
    plt.xlabel("Video")
    plt.ylabel("Reduction (%)")
    plt.title("Compression Percentage per Video")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    bar_chart_path = os.path.join(performance_folder, "reduction_percentage_bar_chart.png")
    plt.savefig(bar_chart_path)
    plt.close()
    print(f"Bar chart saved in: {bar_chart_path}")
    
    print("Performance analysis completed successfully.")

if __name__ == "__main__":
    main()