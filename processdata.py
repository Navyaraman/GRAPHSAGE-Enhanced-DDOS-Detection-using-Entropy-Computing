import pandas as pd
import numpy as np
import json
import logging
from entropy import calculate_column_entropy
import os

# List of columns in your dataset
columns = [
    'scrip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl',
    'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin',
    'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
    'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
    'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
    'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'attack_cat', 'label'
]

# Important attributes for entropy calculation
important_attributes = [
    'sport', 'dsport', 'dur', 'sbytes', 'dbytes', 'Sload', 'Dload',
    'Spkts', 'Dpkts', 'swin', 'dwin', 'trans_depth'
]

# Desired column order for features CSV
desired_feature_column_order = [
    'sport', 'dsport', 'dur', 'sbytes', 'dbytes', 'Sload', 'Dload',
    'Spkts', 'Dpkts', 'swin', 'dwin', 'trans_depth'
]

def is_numeric(value):
    """Check if a value is numeric."""
    try:
        float(value)
        return True
    except ValueError:
        return False

def convert_hex_to_decimal(value):
    """Convert hexadecimal string to decimal if the value is in hexadecimal format."""
    try:
        if isinstance(value, str) and value.startswith('0x'):
            return int(value, 16)
        return float(value)  # Convert to float
    except ValueError:
        return np.nan  # Use NaN if conversion fails

def calculate_min_max_values(df, features_to_scale):
    """Calculate min and max values for scaling."""
    min_max_values = {}
    for feature in features_to_scale:
        if feature in df.columns:
            numeric_values = df[feature].apply(pd.to_numeric, errors='coerce')
            min_val = numeric_values.min()
            max_val = numeric_values.max()
            if min_val != max_val:
                min_max_values[feature] = {'min': float(min_val), 'max': float(max_val)}
    return min_max_values

def process_row(row, min_max_values, features_to_scale):
    """Process a single row for scaling."""
    if isinstance(row, dict):
        row = pd.Series(row) 
    new_row = row.to_dict()
    scaled_row = {}

    for key in desired_feature_column_order:  # Only include desired columns
        try:
            if key in features_to_scale:
                value = convert_hex_to_decimal(new_row.get(key, np.nan))
                
                # Ensure the value is numeric before scaling
                if pd.notna(value):
                    if key in min_max_values:
                        min_val = min_max_values[key]['min']
                        max_val = min_max_values[key]['max']
                        if max_val != min_val:
                            scaled_row[key] = (value - min_val) / (max_val - min_val)
                        else:
                            scaled_row[key] = value
                    else:
                        scaled_row[key] = value
                else:
                    scaled_row[key] = np.nan
            else:
                scaled_row[key] = new_row.get(key, np.nan)
        except (ValueError, KeyError) as e:
            # Handle exceptions related to value conversion and missing keys
            logging.error(f"Error processing row: {e}")
            scaled_row[key] = np.nan

    return scaled_row

def create_edges_from_data(df):
    print("Creating edges from the dataset...")
    print("Number of rows in the DataFrame:", len(df))
    
    # Print some sample rows for debugging
    print("Sample rows:")
    print(df[['scrip', 'dstip']].head(10))
    
    # Create edges based on scrip and dstip
    src_nodes = df['scrip'].tolist()
    dst_nodes = df['dstip'].tolist()
    
    # Print the number of src and dst nodes
    print("Number of src nodes:", len(src_nodes))
    print("Number of dst nodes:", len(dst_nodes))
    
    # Create edges DataFrame
    edges = pd.DataFrame({'src': src_nodes, 'dst': dst_nodes})
    
    # Print the first few rows of edges for debugging
    print("Sample edges:")
    print(edges.head(10))
    
    # Print the total number of edges
    print("Total number of edges:", len(edges))
    
    return edges

def process_large_file(file_path, output_dir, sample_fraction=0.2):
    """Process a large CSV file for batch processing with sampling."""
    print(f"Processing file: {file_path}")
    df = pd.read_csv(file_path, header=None)
    df.columns = columns

    # Sample the dataset
    df_sampled = df.sample(frac=sample_fraction, random_state=42)  # Sample 65% of the data
    print(f"Sampled {len(df_sampled)} rows from the original {len(df)} rows")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Calculating min and max values for scaling...")
    features_to_scale = important_attributes
    min_max_values = calculate_min_max_values(df_sampled, features_to_scale)
    
    # Save min and max values to JSON
    min_max_file = os.path.join(output_dir, 'min_max_values.json')
    with open(min_max_file, 'w') as f:
        json.dump(min_max_values, f, indent=4)
    print(f"Min and max values saved to {min_max_file}")
    
    print("Processing rows...")
    processed_rows = []
    for idx, row in df_sampled.iterrows():
        processed_row = process_row(row, min_max_values, features_to_scale)
        if processed_row:
            processed_rows.append(processed_row)
    print(f"Processed {len(processed_rows)} rows")
    
    if not processed_rows:
        print('No rows processed.')
        return

    print("Calculating column entropy...")
    processed_df = pd.DataFrame(processed_rows)
    column_entropy = calculate_column_entropy(processed_df, important_attributes)
    print(f"Column entropy calculated: {column_entropy}")

    print("Saving features and labels to CSV...")
    features = []
    for row in processed_rows:
        feature_data = {col: row.get(col, np.nan) for col in desired_feature_column_order}
        for attr in important_attributes:
            feature_data[f'{attr}ShannonEntropy'] = column_entropy.get(attr, {}).get('shannon', np.nan)
            feature_data[f'{attr}RenyiEntropy'] = column_entropy.get(attr, {}).get('renyi', np.nan)
        features.append(feature_data)

    features_df = pd.DataFrame(features)
    features_df.to_csv(os.path.join(output_dir, 'features.csv'), index=False)
    print('Features CSV created.')
    
    labels = df_sampled[['label']].copy()  # Directly use the 'label' column from the sampled dataset
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)
    print('Labels CSV created.')

    print("Creating edges...")
    edges_df = create_edges_from_data(df_sampled)
    edges_file = os.path.join(output_dir, 'edges.csv')
    edges_df.to_csv(edges_file, index=False)
    print(f'Edges CSV created at {edges_file}')
