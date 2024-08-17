# evaluate.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from inference import run_inference_and_save

def calculate_metrics(true_labels, predictions, probabilities):
    """Calculate various evaluation metrics."""
    metrics = {}
    
    metrics['Accuracy'] = accuracy_score(true_labels, predictions)
    metrics['Precision'] = precision_score(true_labels, predictions, average='weighted')
    metrics['Recall'] = recall_score(true_labels, predictions, average='weighted')
    metrics['F1 Score'] = f1_score(true_labels, predictions, average='weighted')

    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['True Negatives'] = tn
    metrics['False Positives'] = fp
    metrics['False Negatives'] = fn
    metrics['True Positives'] = tp
    
    metrics['False Positive Rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['False Negative Rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    try:
        metrics['ROC AUC'] = roc_auc_score(true_labels, probabilities)
    except ValueError as e:
        print(f"Warning: ROC AUC calculation failed: {e}")
        metrics['ROC AUC'] = None
    
    try:
        metrics['PR AUC'] = average_precision_score(true_labels, probabilities)
    except ValueError as e:
        print(f"Warning: PR AUC calculation failed: {e}")
        metrics['PR AUC'] = None
    
    return metrics

def plot_precision_recall_curve(true_labels, probabilities):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(true_labels, probabilities)
    
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

def main():
    """Main function to run inference and evaluate."""
    device = torch.device('cpu')  # Use CPU

    test_features_file = 'data/test_features.csv'
    test_edges_file = 'data/test_edges.csv'
    test_labels_file = 'data/test_labels.csv'
    model_path = 'best_model.pth'
    inference_output_file = 'inference_results.csv'

    # Run inference and save results
    run_inference_and_save(model_path, test_features_file, test_edges_file, test_labels_file, inference_output_file, device)

    # Load the inference results
    df = pd.read_csv(inference_output_file)

    true_labels = df['True_Label'].values
    predictions = df['Predicted_Label'].values
    probabilities = df['Probability'].values

    # Calculate metrics
    metrics = calculate_metrics(true_labels, predictions, probabilities)

    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # Plot Precision-Recall Curve
    plot_precision_recall_curve(true_labels, probabilities)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)  # Ignore warnings for undefined metrics
    main()
