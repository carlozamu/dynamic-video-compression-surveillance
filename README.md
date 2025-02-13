# Dynamic Compression for Video Surveillance

**Authors:** Gianluigi Vazzoler & Carlo Zamuner  
**Course:** Signal, Image and Video – UniTn 2024/2025  
**Project Domain:** Video Surveillance & Storage Optimization

---

## Overview

Video surveillance systems produce vast amounts of data, making storage and management challenging—especially in environments with limited hardware or bandwidth. This project introduces a **dynamic compression** system that intelligently distinguishes between high-relevance (motion-intensive) and low-relevance (static) areas in surveillance videos. By applying aggressive compression only to static regions, the system achieves significant file size reduction while preserving critical visual details in motion areas.

The project implements two complementary motion detection techniques:

- **Optical Flow:** Utilizes OpenCV’s `calcOpticalFlowFarneback` to compute motion vectors, generating binary masks that isolate moving areas. This technique offers a good trade-off between processing time and compression quality.
- **Frame Differencing:** Compares consecutive frames to detect pixel-level differences, applying temporal smoothing and morphological filtering to accurately segment static and dynamic regions, albeit with higher computational cost.

A user-friendly graphical interface built with PyQt5 (invoked via `windows.py`) allows users to select input videos, set output directories, choose the processing technique, and even perform an automated performance analysis after conversion.

---

## Project Structure

```
.
├── frame_differencing.py       # Motion detection & compression using frame differencing.
├── motion_compression_opt.py   # Motion detection & compression using optical flow.
├── performance_analysis.py     # Parses logs and generates performance reports/charts.
├── windows.py                  # PyQt5-based GUI for interactive video processing.
├── requirements.txt            # Project dependencies.
└── README.md                   # This file.
```

---

## Installation

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Create a Virtual Environment
#### On Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```
#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Launch the GUI
Start the application using:
```bash
python windows.py
```
The GUI allows you to:
- Select one or multiple input video files.
- Choose the output folder.
- Pick the processing technique (Optical Flow or Frame Differencing).
- Optionally perform performance analysis after processing.
- Monitor real-time logs and processing progress.

### 2. Run Modules via Command Line (only for Debug)
#### Frame Differencing Processing:
```bash
python frame_differencing.py
```
#### Optical Flow Processing:
```bash
python motion_compression_opt.py
```
#### Performance Analysis:
```bash
python performance_analysis.py <output_folder>
```

---

## Reference Paper

### **DYNAMIC COMPRESSION FOR VIDEO SURVEILLANCE**  
**Gianluigi Vazzoler - Carlo Zamuner**  
**Project Course: Signal, Image and Video – UniTn 2024/2025**  

#### **Abstract:**
The paper addresses the challenges of video surveillance data management. It critiques conventional uniform compression methods that often degrade critical details and presents a novel dynamic compression approach. By differentiating between static and motion-intensive areas, the proposed system applies aggressive compression selectively—preserving important visual information while significantly reducing file size. The study evaluates both Optical Flow and Frame Differencing techniques, providing a detailed analysis of their performance trade-offs in terms of quality and processing efficiency.

---

## License

This project is licensed under the MIT License.
