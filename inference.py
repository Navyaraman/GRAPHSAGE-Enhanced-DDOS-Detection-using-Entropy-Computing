# inference.py

import torch
import pandas as pd
import dgl
from model import GraphSAGEGATModel
from dataset import GraphDataset, collate_fn
from trainmodel import create_graph
from torch.utils.data import DataLoader

def load_model(model_path, device):
    """Load the trained model from a checkpoint."""
    model = GraphSAGEGATModel(
        in_feats=36,
        hidden_dim=64,
        num_classes=2,  # Adjust based on the number of classes in your dataset
        num_heads=4
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def create_test_graph(features_file, edges_file):
    """Create a DGL graph from the features and edges files for inference."""
    return create_graph(features_file, edges_file)

def perform_inference(model, test_loader, device):
    """Perform inference on the test data and return predictions, probabilities, and true labels."""
    all_predictions = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_graphs, batch_features, batch_labels in test_loader:
            batch_graphs = batch_graphs.to(device)
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_graphs, batch_features)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])  # Assuming binary classification and interested in the probability of the positive class
            all_labels.extend(batch_labels.cpu().numpy())

    return all_labels, all_predictions, all_probs

def save_predictions(true_labels, predictions, probabilities, output_file):
    """Save the true labels, predictions, and probabilities to a CSV file."""
    df = pd.DataFrame({
        'True_Label': true_labels,
        'Predicted_Label': predictions,
        'Probability': probabilities
    })
    df.to_csv(output_file, index=False)
    print(f"Inference results saved as {output_file}")

def run_inference_and_save(model_path, test_features_file, test_edges_file, test_labels_file, output_file, device):
    """Run inference and save results."""
    model = load_model(model_path, device)
    test_graph = create_test_graph(test_features_file, test_edges_file)
    test_features = torch.tensor(pd.read_csv(test_features_file).values, dtype=torch.float32)
    test_labels = torch.tensor(pd.read_csv(test_labels_file).values.flatten(), dtype=torch.long)

    num_nodes = test_features.shape[0]
    if num_nodes != test_graph.number_of_nodes():
        raise ValueError(f"Number of nodes in the graph ({test_graph.number_of_nodes()}) does not match number of features ({num_nodes})")

    test_dataset = GraphDataset(test_graph, test_features, test_labels, num_nodes_per_batch=3000)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=4)

    true_labels, predictions, probabilities = perform_inference(model, test_loader, device)
    save_predictions(true_labels, predictions, probabilities, output_file)
