# Real-Time Vehicle Detection & Plate Recognition System

This project uses the YOLOv9 model to perform real-time detection of vehicles from a video source (webcam, DroidCam, or video file). It identifies vehicles, determines their color, recognizes license plates, and logs all information into a CSV file.

## Features

- **Real-Time Detection**: Utilizes the high-performance YOLOv9 model
- **Object Identification**: Detects and labels cars, trucks, buses, and motorcycles
- **License Plate Recognition**: Employs EasyOCR to read characters from detected license plates
- **Data Logging**: Records timestamp, location, vehicle type, confidence, color, and plate number
- **User-Friendly GUI**: A simple interface to select the model and video source
- **GPU Accelerated**: Automatically uses CUDA-enabled GPUs if available for faster processing

## Project Structure

```
vehicle_detection_project/
│
├── models/
│   └── yolov9c.pt
├── src/
│   ├── main.py
│   ├── detector.py
│   ├── plate_recognition.py
│   ├── utils.py
│   └── data_logger.py
├── assets/
│   └── test_videos/
├── output/
│   └── logs.csv
├── requirements.txt
├── README.md
└── config.yaml
```

## Setup & Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd vehicle_detection_project
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or if using a custom environment name:

```bash
python -m venv yolo_env
yolo_env\Scripts\activate  # On Windows
source yolo_env/bin/activate  # On Linux/Mac
```

### 3. Install PyTorch with CUDA

Ensure you have a compatible NVIDIA driver installed. Then, install PyTorch using the official command for your CUDA version.

**For CUDA 11.8:**

```bash
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**For other CUDA versions**, visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to get the appropriate installation command.

### 4. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download YOLOv9 Model

1. Create a `models` directory if it doesn't exist
2. Download the `yolov9c.pt` weights from the [official YOLOv9 repository](https://github.com/WongKinYiu/yolov9) or another trusted source
3. Place the model file inside the `models` folder

## How to Run

### Method 1: Using the GUI

Execute the `main.py` script to launch the GUI:

```bash
python src/main.py
```

Or with the virtual environment:

```bash
yolo_env\Scripts\activate
python -m src.main
```

### From the GUI:

1. Click **"Select Model"** to choose your `.pt` model file (e.g., `models/yolov9c.pt`)
2. Enter the video source in the text box. This can be:
   - A camera index (e.g., `0` for the default webcam, `1` or `2` for DroidCam)
   - A path to a video file (e.g., `assets/test_videos/traffic.mp4`)
3. Click **"Start Detection"**
4. A window will appear showing the real-time detection feed
5. Press the **'q'** key to stop the detection and close the window

## Quick Start Commands

```bash
# Activate the virtual environment
yolo_env\Scripts\activate

# Run the application
python -m src.main
```

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for real-time performance)
- Webcam or video file for testing

## Output

All detection logs are saved to `output/logs.csv` with the following information:
- Timestamp
- Location coordinates
- Vehicle type
- Detection confidence
- Vehicle color
- License plate number (if detected)

## Troubleshooting

### GPU Not Detected
- Ensure you have the correct NVIDIA drivers installed
- Verify CUDA installation with: `nvidia-smi`
- Check PyTorch CUDA availability in Python:
  ```python
  import torch
  print(torch.cuda.is_available())
  ```

### Camera Not Working
- Try different camera indices (0, 1, 2)
- Ensure no other application is using the camera
- Check camera permissions in your OS settings

### Model Loading Issues
- Verify the model file path is correct
- Ensure the model file is not corrupted
- Check that you have sufficient disk space

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]



NOTE: Under the "models" folder, download the the existing yolo models from the website.

## Acknowledgments

- YOLOv9 by WongKinYiu
- EasyOCR for optical character recognition
- PyTorch team for the deep learning framework
