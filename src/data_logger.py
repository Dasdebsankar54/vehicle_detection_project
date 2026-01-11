import csv
import os
from threading import Lock

class DataLogger:
    """
    A thread-safe class to handle logging detection data to a CSV file.
    """
    def __init__(self, file_path, header):
        """
        Initializes the DataLogger.

        Args:
            file_path (str): The full path to the CSV log file.
            header (list): A list of strings for the CSV header.
        """
        self.file_path = file_path
        self.header = header
        self.lock = Lock()  # To prevent race conditions when writing from multiple threads

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write the header if the file is new/empty
        self._initialize_file()

    def _initialize_file(self):
        """
        Writes the header to the CSV file if it doesn't exist or is empty.
        """
        with self.lock:
            # Check if file exists and is not empty
            file_exists = os.path.isfile(self.file_path) and os.path.getsize(self.file_path) > 0
            if not file_exists:
                try:
                    with open(self.file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(self.header)
                except IOError as e:
                    print(f"Error initializing log file: {e}")

    def log(self, data):
        """
        Appends a new row of data to the CSV file.

        Args:
            data (dict): A dictionary where keys match the header fields.
        """
        with self.lock:
            try:
                # Create a list of values in the correct order based on the header
                row = [data.get(h, '') for h in self.header]

                with open(self.file_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

            except IOError as e:
                print(f"Error writing to log file: {e}")
            except Exception as e:
                print(f"An unexpected error occurred during logging: {e}")

