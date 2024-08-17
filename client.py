import pandas as pd
import requests
import logging
import numpy as np
from tqdm import tqdm  # For progress bar

# Configure logging
logging.basicConfig(filename='data_sending.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the local server URL
SERVER_URL = 'http://127.0.0.1:5001/upload'

# Define dataset file path
DATASET_FILE = 'sampledata.csv'

def clamp_float(value, min_val=-1e+308, max_val=1e+308):
    """Clamp float values to a range that is JSON-compliant."""
    if isinstance(value, float):
        if np.isnan(value):
            return 0  # Replace NaN with 0
        if value > max_val or value < min_val:
            logging.warning(f"Value {value} out of range, clamping to {max_val if value > max_val else min_val}")
            return max(min(value, max_val), min_val)
        return value
    return value

def sanitize_data(data):
    """Sanitize data to ensure JSON compliance."""
    sanitized_data = []
    for record in data:
        sanitized_record = {}
        for key, value in record.items():
            sanitized_record[key] = clamp_float(value)
        sanitized_data.append(sanitized_record)
    return sanitized_data

def read_csv_as_json(file_path):
    """Read CSV file and convert it to JSON format."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        data = df.to_dict(orient='records')
        sanitized_data = sanitize_data(data)
        logging.info(f"Sanitized Data: {sanitized_data[:5]}")  # Log first 5 records for inspection
        return sanitized_data
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return []

def send_data_to_server(json_data):
    """Send JSON data to the server."""
    try:
        response = requests.post(SERVER_URL, json=json_data, headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        logging.info(f"Server response status code: {response.status_code}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending data to server: {e}")
        return {"error": str(e)}

def process_dataset():
    """Process the dataset and send it to the server."""
    json_data = read_csv_as_json(DATASET_FILE)
    
    if json_data:
        # Ensure data is a list
        if not isinstance(json_data, list):
            logging.error("Data is not in list format.")
            return
        
        response = send_data_to_server(json_data)
        logging.info(f"Server response: {response}")
    else:
        logging.error("No data to send.")

if __name__ == "__main__":
    process_dataset()
