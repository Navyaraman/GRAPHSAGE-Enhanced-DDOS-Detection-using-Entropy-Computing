import os
from processdata import process_large_file

# Path to your original dataset
original_dataset_path = 'UNSW-NB15_4.csv'
# Directory to save the processed data and calculated values
output_dir = 'data'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

print(f"Starting processing for {original_dataset_path}...")

try:
    # Process the large dataset
    process_large_file(original_dataset_path, output_dir)
    print("Processing completed.")
except Exception as e:
    print(f"An error occurred: {e}")
