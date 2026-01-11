import cv2
import easyocr
import numpy as np
import re
import yaml

class PlateRecognizer:
    """
    Handles license plate detection and OCR with improved robustness and a visual debugger.
    This version combines text fragments to handle multi-part license plates.
    """
    def __init__(self, languages=['en'], config_path='config.yaml', debug=False):
        """
        Initializes the OCR reader, loads config, and sets debug mode.
        """
        print("Initializing EasyOCR... This may take a moment.")
        self.reader = easyocr.Reader(languages, gpu=True)
        print("EasyOCR initialized.")
        self.debug = debug

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.min_aspect_ratio = config.get('plate_min_aspect_ratio', 2.0)
            self.max_aspect_ratio = config.get('plate_max_aspect_ratio', 5.0)
            self.min_area = config.get('plate_min_area', 400)

    def _visualize_step(self, title, image):
        """Displays an image if debug mode is active."""
        if self.debug:
            # Resize for better viewing if the image is too large
            h, w = image.shape[:2]
            if h > 800 or w > 800:
                scale = 800 / max(h, w)
                image = cv2.resize(image, (int(w*scale), int(h*scale)))
            cv2.imshow(title, image)

    def _find_plate_candidates(self, vehicle_image):
        """
        Finds potential license plate regions in a vehicle image using contour detection.
        """
        if vehicle_image is None or vehicle_image.size == 0:
            return []

        gray = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2GRAY)
        if self.debug: self._visualize_step("1. Grayscale", gray)

        # Increase kernel size for more aggressive blackhat to find larger text groups
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
        if self.debug: self._visualize_step("2. Blackhat", blackhat)

        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        (min_val, max_val) = (np.min(grad_x), np.max(grad_x))
        grad_x = 255 * ((grad_x - min_val) / (max_val - min_val)) if max_val > min_val else 0
        grad_x = grad_x.astype("uint8")
        if self.debug: self._visualize_step("3. Sobel Gradient", grad_x)

        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
        thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        if self.debug: self._visualize_step("4. Thresholded", thresh)

        # Add more dilation to connect text parts
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        if self.debug: self._visualize_step("5. Dilated/Eroded", thresh)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            if self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio and w * h > self.min_area:
                candidates.append(c)
        
        if self.debug and candidates:
            debug_img = vehicle_image.copy()
            cv2.drawContours(debug_img, candidates, -1, (0, 255, 0), 2)
            self._visualize_step("6. Plate Candidates", debug_img)

        return candidates

    def _cleanup_text(self, text):
        """Cleans the OCR'd text."""
        return "".join(re.findall(r"[A-Z0-9]", text)).upper()

    def find_and_read_plate(self, vehicle_image, plate_confidence_threshold=0.5):
        """
        Finds and reads the license plate by checking all valid candidates and combining
        text fragments found within the best candidate.
        """
        plate_contours = self._find_plate_candidates(vehicle_image)

        best_plate_text = ""
        highest_avg_confidence = 0.0

        for contour in plate_contours:
            x, y, w, h = cv2.boundingRect(contour)
            plate_crop = vehicle_image[y:y+h, x:x+w]

            if plate_crop.size == 0:
                continue

            results = self.reader.readtext(plate_crop)
            if not results:
                continue

            # --- NEW LOGIC: Combine text fragments ---
            # 1. Filter results by confidence
            confident_results = [res for res in results if res[2] > plate_confidence_threshold]
            if not confident_results:
                continue
            
            # 2. Sort fragments by their horizontal position (left to right)
            confident_results.sort(key=lambda r: r[0][0][0]) # Sort by the x-coordinate of the top-left point
            
            # 3. Combine the text and calculate average confidence
            combined_text = " ".join([self._cleanup_text(res[1]) for res in confident_results])
            avg_confidence = sum([res[2] for res in confident_results]) / len(confident_results)
            
            # Check if this combined result is better than what we've seen so far
            if avg_confidence > highest_avg_confidence:
                highest_avg_confidence = avg_confidence
                best_plate_text = combined_text

        # If in debug mode, pause execution
        if self.debug and plate_contours:
            print(f"Debug - Best plate found: '{best_plate_text}' with avg confidence {highest_avg_confidence:.2f}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if best_plate_text:
            return best_plate_text, highest_avg_confidence

        return None, 0.0
