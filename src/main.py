import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import cv2
import yaml

# --- Pre-emptive Check for NumPy ---
try:
    import numpy as np
except ImportError:
    messagebox.showerror(
        "Dependency Error",
        "NumPy is not installed. Please run 'pip install numpy'."
    )
    sys.exit(1)

from src.detector import VehicleDetector  # Ensure this exists

class Application(tk.Tk):
    """
    The main GUI application window for the detection system.
    """
    def __init__(self):
        super().__init__()
        self.title("YOLOv9 Vehicle Detection Setup")
        self.geometry("550x350")

        # Load default config
        try:
            with open('config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            messagebox.showerror("Error", "config.yaml not found! Please keep it in the root folder.")
            self.destroy()
            return

        self.model_path = tk.StringVar()
        self.video_source = tk.StringVar()

        self.create_widgets()
        self.populate_video_sources()

    def get_available_cameras(self):
        """
        Detects available camera indices by testing OpenCV video capture.
        Returns:
            list: A list of detected camera indices.
        """
        available = []
        for i in range(10):  # Try first 10 indices
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                available.append(i)
            cap.release()
        return available

    def create_widgets(self):
        """Creates and places all the widgets in the main window."""
        main_frame = tk.Frame(self, padx=20, pady=20)
        main_frame.pack(expand=True, fill=tk.BOTH)

        # --- Model Selection ---
        model_frame = tk.LabelFrame(main_frame, text="1. Select YOLOv9 Model", padx=10, pady=10)
        model_frame.pack(fill=tk.X, pady=10)

        model_entry = tk.Entry(model_frame, textvariable=self.model_path, state='readonly', width=50)
        model_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 10))

        model_button = tk.Button(model_frame, text="Browse...", command=self.select_model_file)
        model_button.pack(side=tk.RIGHT)

        # --- Video Source Selection ---
        source_frame = tk.LabelFrame(main_frame, text="2. Select or Enter Video Source", padx=10, pady=10)
        source_frame.pack(fill=tk.X, pady=10)

        source_label = tk.Label(source_frame, text="Available Cameras or Path to Video:")
        source_label.pack(anchor='w')

        self.source_combobox = ttk.Combobox(source_frame, textvariable=self.video_source, width=58)
        self.source_combobox.pack(fill=tk.X, pady=(5, 0))

        # --- Start Button ---
        start_button = tk.Button(main_frame, text="Start Detection", bg="#4CAF50", fg="white",
                                 font=('Helvetica', 12, 'bold'), command=self.start_detection)
        start_button.pack(pady=20, ipady=10, fill=tk.X)

    def populate_video_sources(self):
        """Detects and fills the combobox with camera sources."""
        print("Scanning for available video devices...")
        cameras = self.get_available_cameras()
        camera_options = [f"Camera {i}" for i in cameras]

        if not camera_options:
            print("No cameras found.")
            camera_options.append("No cameras detected")
        else:
            print(f"Found cameras: {cameras}")

        self.source_combobox['values'] = camera_options

        default_source = str(self.config.get('default_video_source', '0'))
        if f"Camera {default_source}" in camera_options:
            self.video_source.set(f"Camera {default_source}")
        elif camera_options:
            self.video_source.set(camera_options[0])

    def select_model_file(self):
        """Opens file dialog to choose a YOLO model file."""
        initial_dir = os.path.abspath('models')
        os.makedirs(initial_dir, exist_ok=True)

        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select YOLOv9 Model File",
            filetypes=(("PyTorch Models", "*.pt"), ("All files", "*.*"))
        )
        if file_path:
            self.model_path.set(file_path)

    def start_detection(self):
        """Starts detection after validating inputs."""
        model = self.model_path.get()
        source_input = self.video_source.get()

        if not model:
            messagebox.showwarning("Input Error", "Please select a YOLOv9 model file.")
            return

        if not source_input.strip() or "No cameras detected" in source_input:
            messagebox.showwarning("Input Error", "Please select a valid video source.")
            return

        # Determine video source
        if "Camera" in source_input:
            source = int(source_input.split()[-1])
        else:
            source = source_input

        try:
            self.withdraw()
            detector = VehicleDetector(model_path=model, config_path='config.yaml')
            detector.run_detection(video_source=source)
        except Exception as e:
            messagebox.showerror("Runtime Error", f"An error occurred: {e}")
        finally:
            self.deiconify()


if __name__ == "__main__":
    app = Application()
    app.mainloop()
