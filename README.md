# Dynamic Video Compression for Surveillance Systems

## Overview
This project implements a dynamic video compression system designed specifically for surveillance video streams. The system leverages motion detection techniques to preserve high quality in critical areas while applying advanced compression to static regions, optimizing storage and bandwidth without compromising essential information.

## Features
- **Motion Detection:** Identifies areas of movement using optical flow techniques.
- **Selective Compression:** Maintains high quality in regions of interest and applies aggressive compression to static areas.
- **Resource Optimization:** Reduces file size and storage requirements while retaining critical visual data.

## Technologies Used
- Python
- OpenCV
- Numpy
- Custom functions for quantization and compression (e.g., zigzag, run-length encoding).

## How It Works
1. **Motion Estimation:** Analyzes video frames to generate motion masks using optical flow.
2. **Compression:** Applies JPEG-like quantization selectively based on motion masks:
   - High quality for moving regions.
   - Aggressive compression for static regions.
3. **Output Generation:** Reconstructs the optimized video for efficient storage and transmission.

## Usage
1. Place input videos (`.mp4` format) in the `input` directory.
2. Run the script to process all videos in the input folder:
   ```bash
   python main.py
   ```
3. Optimized videos will be saved in the `output` directory.

## Repository Structure
```
.
├── input/                  # Directory for input videos
├── output/                 # Directory for compressed videos
├── functions.py            # Utility functions for compression and encoding
├── main.py                 # Main script for video processing
└── README.md               # Project documentation
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contributors
- Gianluigi Vazzoler
- Carlo Zamuner

## Acknowledgments
Special thanks to the Signal Image and Video course for foundational concepts and guidance.
