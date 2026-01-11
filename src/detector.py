import cv2
import yaml
import datetime
import geocoder
from ultralytics import YOLO

from src.data_logger import DataLogger
from src.plate_recognition import PlateRecognizer
from src.utils import get_dominant_color, get_color_name, get_timestamp

class VehicleDetector:
    """
    Manages the vehicle detection process using a YOLO model.
    """
    def __init__(self, model_path, config_path='config.yaml'):
        """
        Initializes the detector with the model and configuration.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("Fetching real-time location based on IP address...")
        try:
            g = geocoder.ip('me')
            self.location = f"{g.city}, {g.country}" if g.ok else self.config.get('location', 'Unknown Location')
        except Exception:
            self.location = self.config.get('location', 'Error-Fetching-Location')
        print(f"Location set to: {self.location}")

        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        print("YOLO model loaded successfully.")

        self.class_names = self.model.names
        self.target_classes = self.config['target_classes']
        self.color_map = {
            'car': (255, 0, 0), 'truck': (0, 0, 255), 'bus': (0, 255, 255),
            'motorcycle': (0, 255, 0), 'person': (255, 0, 255)
        }

        # --- Pass the debug flag to the PlateRecognizer ---
        self.plate_recognizer = PlateRecognizer(
            languages=self.config['ocr_languages'],
            config_path=config_path,
            debug=self.config.get('debug_plate_recognition', False)
        )

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"detection_log_{timestamp_str}.csv"
        log_file_path = f"{self.config['output_directory']}/{log_filename}"
        print(f"Logging detections to: {log_file_path}")
        self.data_logger = DataLogger(file_path=log_file_path, header=self.config['log_header'])

    def process_frame(self, frame):
        results = self.model(frame, conf=self.config['confidence_threshold'], iou=self.config['iou_threshold'], classes=self.target_classes, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                class_name = self.class_names[cls_id]
                confidence = float(box.conf[0])
                vehicle_crop = frame[y1:y2, x1:x2]

                color_name = get_color_name(get_dominant_color(vehicle_crop)).capitalize()
                plate_text, plate_confidence = self.plate_recognizer.find_and_read_plate(
                    vehicle_crop, self.config['plate_confidence_threshold']
                )

                log_data = {
                    "timestamp": get_timestamp(), "location": self.location, "class_name": class_name,
                    "confidence_score": f"{confidence:.2f}", "vehicle_color": color_name,
                    "license_plate": plate_text if plate_text else "N/A",
                    "plate_confidence": f"{plate_confidence:.2f}" if plate_text else "N/A"
                }
                self.data_logger.log(log_data)

                box_color = self.color_map.get(class_name.lower(), (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        return frame

    def run_detection(self, video_source):
        try:
            video_source = int(video_source)
        except ValueError:
            pass
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source '{video_source}'")
            return
        while True:
            success, frame = cap.read()
            if not success:
                break
            processed_frame = self.process_frame(frame)
            cv2.imshow(self.config['project_name'], processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")
