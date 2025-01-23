# Dynamic Video Compression with Motion-Based Processing

This project implements a motion-based video compression system using Python, OpenCV, and a graphical user interface (GUI) powered by PySimpleGUI. The system identifies areas of motion in a video and applies different levels of compression to static and dynamic regions, resulting in efficient storage without significant quality loss in areas of interest.

## Features
- **Motion Detection:** Uses optical flow to detect areas of motion in videos.
- **Dynamic Compression:** Compresses static areas more aggressively while preserving quality in regions with motion.
- **Graphical User Interface (GUI):** Users can select input videos and output directories through a simple and intuitive GUI.
- **Real-Time Logging:** Displays processing logs in real-time within the GUI.
- **Batch Processing Support:** Processes multiple videos in the `Dataset/input/` folder when run in batch mode.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/dynamic-video-compression.git
   cd dynamic-video-compression
   ```

2. **Install Dependencies**
   Ensure you have Python 3.x installed. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   The main dependencies include:
   - `opencv-python`
   - `numpy`
   - `PySimpleGUI`

## Usage

### Graphical User Interface (GUI)
The GUI allows users to process videos interactively.

1. Run the `windows.py` script:
   ```bash
   python windows.py
   ```

2. Select the input video file and output directory using the file and folder browser.

3. Click the **"Avvia"** button to start the processing.

4. Observe the logs in the GUI to monitor the progress. Once completed, the results will be saved in the selected output directory.

### Batch Processing (Command Line)
To process all `.mp4` files in the `Dataset/input/` folder and save results in `Dataset/output/`:

1. Place the videos to be processed in the `Dataset/input/` directory.
2. Run the `motion_compression.py` script:
   ```bash
   python motion_compression.py
   ```
3. Processed videos will be saved in `Dataset/output/` with subdirectories for each input file.

## Output Files
For each input video, the following files are generated:
- **`overlay.mp4`**: A copy of the video with motion regions highlighted.
- **`mask.mp4`**: A binary mask video showing the detected motion areas.
- **`compressed.mp4`**: The final compressed video with motion-aware adjustments.

## Code Structure

```
.
├── motion_compression.py   # Main processing script
├── windows.py              # GUI for user interaction
├── Dataset/
│   ├── input/              # Folder for input videos
│   ├── output/             # Folder for output results
├── requirements.txt        # Required Python dependencies
└── README.md               # Project documentation
```

## Key Functions

### `motion_compression.py`
- **`setup_logging(output_dir)`**: Configures logging for real-time feedback.
- **`temporal_smoothing_flow(video_path, output_dir, ...)`**: Detects motion using optical flow and generates mask and overlay videos.
- **`compress_with_motion(input_video, mask_video, output_dir)`**: Compresses static and dynamic regions differently.
- **`process_single_video(video_path, output_dir)`**: Combines all processing steps for a single video.

### `windows.py`
- Implements the GUI for video selection and output management.
- Displays logs in real-time within the application.

## Example Workflow

1. Start the GUI:
   ```bash
   python windows.py
   ```

2. Select a video file and an output directory.

3. Click **"Avvia"** to begin processing. Logs will display the progress.

4. Check the output directory for the resulting videos (`overlay.mp4`, `mask.mp4`, `compressed.mp4`).

5. For batch processing, place videos in `Dataset/input/` and run:
   ```bash
   python motion_compression.py
   ```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributions
Feel free to submit issues or pull requests to improve the project. Feedback is welcome!
