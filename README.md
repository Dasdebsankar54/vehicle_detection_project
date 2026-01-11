Real-Time Vehicle Detection & Plate Recognition System
This project uses the YOLOv9 model to perform real-time detection of vehicles from a video source (webcam, DroidCam, or video file). It identifies vehicles, determines their color, recognizes license plates, and logs all information into a CSV file.

Features
Real-Time Detection: Utilizes the high-performance YOLOv9 model.

Object Identification: Detects and labels cars, trucks, buses, and motorcycles.

License Plate Recognition: Employs EasyOCR to read characters from detected license plates.

Data Logging: Records timestamp, location, vehicle type, confidence, color, and plate number.

User-Friendly GUI: A simple interface to select the model and video source.

GPU Accelerated: Automatically uses CUDA-enabled GPUs if available for faster processing.

Project Structure
vehicle_detection_project/
│
├── models/
│ └── yolov9c.pt
├── src/
│ ├── main.py
│ ├── detector.py
│ ├── plate_recognition.py
│ ├── utils.py
│ └── data_logger.py
├── assets/
│ └── test_videos/
├── output/
│ └── logs.csv
├── requirements.txt
├── README.md
└── config.yaml

Setup & Installation
Clone the Repository:

git clone <your-repo-url>
cd vehicle_detection_project

Create a Virtual Environment (Recommended):

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install PyTorch with CUDA:
Ensure you have a compatible NVIDIA driver installed. Then, install PyTorch using the official command for your CUDA version. For CUDA 11.8:

pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 --index-url https://download.pytorch.org/whl/cu118

Install Other Dependencies:

pip install -r requirements.txt

Download YOLOv9 Model:

Create a models directory.

Download the yolov9c.pt weights from the official YOLOv9 repository or another trusted source and place it inside the models folder.

How to Run
Execute the main.py script to launch the GUI:

python src/main.py

From the GUI:

Click "Select Model" to choose your .pt model file (e.g., models/yolov9c.pt).

Enter the video source in the text box. This can be:

A camera index (e.g., 0 for the default webcam, 1 or 2 for DroidCam).

A path to a video file (e.g., assets/test_videos/traffic.mp4).

Click "Start Detection".

A window will appear showing the real-time detection feed. Press the 'q' key to stop the detection and close the window.

To run this file:

yolo_env\Scripts\activate

python -m src.main
