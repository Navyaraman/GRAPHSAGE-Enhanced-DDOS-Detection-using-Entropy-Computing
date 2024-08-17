import torch
import pandas as pd
import numpy as np
import json
import logging
import os
import tempfile
from flask import Flask, request, jsonify
from model import GraphSAGEGATModel
from processdata import process_row, create_edges_from_data
from trainmodel import create_graph
from entropy import calculate_column_entropy

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
def load_model(model_path='best_model.pth'):
    model = GraphSAGEGATModel(
        in_feats=36,  # Update with actual input feature dimension
        hidden_dim=64,
        num_classes=2,  # Update with actual number of classes
        num_heads=4
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Load the min-max values for scaling
def load_min_max_values(min_max_path='data/min_max_values.json'):
    with open(min_max_path, 'r') as f:
        return json.load(f)

# Load the trained model and min-max values
model = load_model()
min_max_values = load_min_max_values()

# Define data preprocessing and prediction functions
def preprocess_data(data):
    # Convert incoming JSON data to DataFrame
    df = pd.DataFrame(data)
    
    # Process and scale the features
    try:
        important_attributes = [
            'sport', 'dsport', 'dur', 'sbytes', 'dbytes', 'Sload', 'Dload',
            'Spkts', 'Dpkts', 'swin', 'dwin', 'trans_depth'
        ]
        processed_df = df.apply(lambda row: process_row(row, min_max_values, important_attributes), axis=1)
        processed_df = pd.DataFrame(processed_df.tolist())
    except Exception as e:
        logging.error(f"Error in process_row: {e}")
        raise

    # Calculate entropy for the features
    try:
        column_entropy = calculate_column_entropy(processed_df, important_attributes)
        logging.info(f"Column entropy calculated: {column_entropy}")
    except Exception as e:
        logging.error(f"Error calculating entropy: {e}")
        raise

    # Add entropy values to the processed row
    try:
        for attr in important_attributes:
            processed_df[f'{attr}ShannonEntropy'] = column_entropy.get(attr, {}).get('shannon', np.nan)
            processed_df[f'{attr}RenyiEntropy'] = column_entropy.get(attr, {}).get('renyi', np.nan)
    except Exception as e:
        logging.error(f"Error adding entropy values: {e}")
        raise

    # Create edges from the data
    try:
        edges = create_edges_from_data(df)
        logging.info(f"Edges created: {edges}")
    except Exception as e:
        logging.error(f"Error in create_edges_from_data: {e}")
        raise

    # Save features and edges to temporary CSV files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_features_file, \
         tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_edges_file:
        features_file = temp_features_file.name
        edges_file = temp_edges_file.name

        processed_df.to_csv(features_file, index=False)
        edges.to_csv(edges_file, index=False)

    # Create graph from features and edges
    try:
        graph = create_graph(features_file, edges_file)
        return graph, torch.tensor(processed_df.values, dtype=torch.float32)
    except Exception as e:
        logging.error(f"Error in create_graph: {e}")
        raise

def perform_inference(model, graph, features, device):
    """Perform inference on the data and return predictions and probabilities."""
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        graph = graph.to(device)
        features = features.to(device)
        
        # Forward pass
        outputs = model(graph, features)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        # Extract probabilities for the positive class
        all_predictions = predicted.cpu().numpy().tolist()
        all_probs = probs.cpu().numpy()[:, 1].tolist()  # Assuming binary classification
        
    return all_predictions, all_probs

def make_prediction(graph, features):
    device = torch.device('cpu')
    predictions, probabilities = perform_inference(model, graph, features, device)
    return predictions, probabilities

@app.route('/upload', methods=['POST'])
def upload_data():
    try:
        if not request.is_json:
            logging.error("Request is not JSON")
            raise ValueError("Request must be in JSON format")
        
        data = request.get_json()
        
        if not isinstance(data, list):
            logging.error(f"Expected list, but got {type(data)}")
            raise ValueError("Expected a list of records as JSON array.")
        
        # Save raw incoming data to CSV
        save_raw_data_to_csv(data)

        # Process and predict
        try:
            graph, features = preprocess_data(data)
            predictions, probabilities = make_prediction(graph, features)
            results = [{'prediction': int(pred), 'probability': prob} for pred, prob in zip(predictions, probabilities)]
        except Exception as e:
            logging.error(f"Error processing data: {e}")
            results = [{'record': record, 'error': str(e)} for record in data]

        # Save predictions to CSV
        save_predictions_to_csv(results)

        return jsonify({"status": "success", "message": "Data processed and predictions saved.", "results": results})
    
    except ValueError as e:
        logging.error(f"ValueError: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"Error processing data: {e}")
        return jsonify({"error": str(e)}), 500

def save_raw_data_to_csv(data, file_path='data/raw_data.csv'):
    """Save the raw incoming data to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    
def save_predictions_to_csv(results, file_path='data/predictions.csv'):
    """Save the predictions to a CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)
    logging.info(f"Predictions saved to {file_path}")

if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')
    app.run(port=5001, debug=True)
