import datetime
import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_timestamp():
    """
    Returns the current timestamp in a standard string format.

    Returns:
        str: Formatted timestamp (e.g., "YYYY-MM-DD_HH-MM-SS").
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_dominant_color(image, k=3):
    """
    Finds the dominant color in an image using K-Means clustering.

    Args:
        image (np.ndarray): The input image (cropped to the detected object).
        k (int): The number of clusters for K-Means. Defaults to 3.

    Returns:
        tuple: A tuple representing the dominant BGR color.
               Returns (0, 0, 0) (black) if the image is invalid.
    """
    try:
        # Ensure the image is valid
        if image is None or image.size == 0:
            return (0, 0, 0)

        # Reshape the image to be a list of pixels
        # The image should be in BGR format from OpenCV
        image = image.reshape((image.shape[0] * image.shape[1], 3))

        # Use K-Means to find the most common color
        clt = KMeans(n_clusters=k, n_init=10)
        clt.fit(image)

        # The cluster center with the most pixels is the dominant color
        num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=num_labels)

        # Find the most frequent label
        hist = hist.astype("float")
        hist /= hist.sum()

        # Get the color of the most frequent cluster
        dominant_color = clt.cluster_centers_[np.argmax(hist)]

        return tuple(map(int, dominant_color))

    except Exception as e:
        print(f"Error in color detection: {e}")
        return (0, 0, 0) # Return black color in case of an error

def get_color_name(bgr_tuple):
    """
    Converts a BGR color tuple to a simple color name.
    This is a very basic implementation. For more accuracy, a more
    sophisticated color mapping library or method would be needed.

    Args:
        bgr_tuple (tuple): A tuple of (B, G, R) values.

    Returns:
        str: The name of the color (e.g., 'black', 'white', 'red').
    """
    # Define color ranges in BGR
    # Note: This is a simplified mapping and may not be highly accurate.
    colors = {
        "black": ([0, 0, 0], [70, 70, 70]),
        "white": ([200, 200, 200], [255, 255, 255]),
        "red": ([0, 0, 100], [80, 80, 255]),
        "green": ([0, 100, 0], [80, 255, 80]),
        "blue": ([100, 0, 0], [255, 80, 80]),
        "yellow": ([0, 180, 180], [100, 255, 255]),
        "gray": ([71, 71, 71], [199, 199, 199])
    }
    
    # Check for black and white first as they are common
    if bgr_tuple[0] < 75 and bgr_tuple[1] < 75 and bgr_tuple[2] < 75:
        return "black"
    if bgr_tuple[0] > 190 and bgr_tuple[1] > 190 and bgr_tuple[2] > 190:
        return "white"

    # Simplified hue-based check
    b, g, r = [x / 255.0 for x in bgr_tuple]
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    delta = max_val - min_val

    if delta == 0:
        return "gray"

    if max_val == r:
        return "red"
    elif max_val == g:
        return "green"
    else:
        return "blue"

    return "unknown"
