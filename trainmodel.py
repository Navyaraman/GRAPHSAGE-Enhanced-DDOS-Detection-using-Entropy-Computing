import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import pickle
import dgl
from model import GraphSAGEGATModel
from dataset import GraphDataset, collate_fn
from torch.optim.lr_scheduler import StepLR
import gc

# Set environment variables for large storage
os.environ['TORCH_HOME'] = 'D:/torch_home'
os.environ['TMPDIR'] = 'D:/temp'



def split_edges(edges_file, train_edges_file, val_edges_file, test_edges_file, test_size=0.2, val_size=0.1):
    """Split the edges file into train, validation, and test sets."""
    edges_df = pd.read_csv(edges_file, header=0, low_memory=False)
    
    # Shuffle the edges
    edges_df = edges_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split the edges into training, validation, and test sets
    num_edges = len(edges_df)
    test_split = int(num_edges * test_size)
    val_split = int(num_edges * val_size)
    
    test_edges_df = edges_df[:test_split]
    val_edges_df = edges_df[test_split:test_split + val_split]
    train_edges_df = edges_df[test_split + val_split:]
    
    test_edges_df.to_csv(test_edges_file, index=False)
    val_edges_df.to_csv(val_edges_file, index=False)
    train_edges_df.to_csv(train_edges_file, index=False)
    
    print("Edge files split and saved.")
    print(f"Total edges: {num_edges}")
    print(f"Training edges: {len(train_edges_df)}")
    print(f"Validation edges: {len(val_edges_df)}")
    print(f"Test edges: {len(test_edges_df)}")
def split_and_save_data(features_file, labels_file, edges_file, train_features_file, val_features_file, test_features_file, train_labels_file, val_labels_file, test_labels_file, train_edges_file, val_edges_file, test_edges_file, test_size=0.2, val_size=0.1):
    print("Starting data split and save...")
    
    # Load features and labels
    features = pd.read_csv(features_file, header=0, low_memory=False, dtype=np.float32)
    labels = pd.read_csv(labels_file, header=0).iloc[:, 0].values
    
    print("Initial features columns:", features.columns.tolist())
    
    # Encode labels
    labels[labels == 2] = 1
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    print("Unique labels after encoding:", np.unique(labels_encoded))
    
    # Separate numeric and non-numeric features
    numeric_features = features.select_dtypes(include=[np.number])
    non_numeric_features = features.select_dtypes(exclude=[np.number])
    
    print("Numeric features columns:", numeric_features.columns.tolist())
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    numeric_features_imputed = imputer.fit_transform(numeric_features)
    
    # Encode non-numeric features
    label_encoders = {}
    for col in non_numeric_features.columns:
        le = LabelEncoder()
        non_numeric_features[col] = le.fit_transform(non_numeric_features[col].astype(str))
        label_encoders[col] = le

    print("Non-numeric features columns after encoding:", non_numeric_features.columns.tolist())
    
    # Concatenate features
    all_features = np.hstack([numeric_features_imputed, non_numeric_features])
    all_features_columns = numeric_features.columns.tolist() + non_numeric_features.columns.tolist()
    
    print("All features columns after concatenation:", all_features_columns)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(all_features, labels_encoded, test_size=test_size + val_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Resample training data
    smote = SMOTE(random_state=42, k_neighbors=4)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(train_features_file), exist_ok=True)
    os.makedirs(os.path.dirname(val_features_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_features_file), exist_ok=True)
    os.makedirs(os.path.dirname(train_labels_file), exist_ok=True)
    os.makedirs(os.path.dirname(val_labels_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_labels_file), exist_ok=True)
    os.makedirs(os.path.dirname(train_edges_file), exist_ok=True)
    os.makedirs(os.path.dirname(val_edges_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_edges_file), exist_ok=True)

    # Generate and save edge files
    print("Generating edge files...")
    split_edges(edges_file, train_edges_file, val_edges_file, test_edges_file, test_size, val_size)

    # Save features and labels
    pd.DataFrame(X_train_resampled, columns=all_features_columns).to_csv(train_features_file, index=False)
    pd.DataFrame(X_val_scaled, columns=all_features_columns).to_csv(val_features_file, index=False)
    pd.DataFrame(X_test_scaled, columns=all_features_columns).to_csv(test_features_file, index=False)
    pd.DataFrame(y_train_resampled).to_csv(train_labels_file, index=False)
    pd.DataFrame(y_val).to_csv(val_labels_file, index=False)
    pd.DataFrame(y_test).to_csv(test_labels_file, index=False)

    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    print("Dataset split and saved successfully.")
    print(f"Training features shape: {X_train_resampled.shape}")
    print(f"Validation features shape: {X_val_scaled.shape}")
    print(f"Test features shape: {X_test_scaled.shape}")



def create_graph(features_file, edges_file, max_nodes=None):
    features_df = pd.read_csv(features_file, header=0, low_memory=False)
    features_df = features_df.apply(pd.to_numeric, errors='coerce').dropna()
    features = torch.tensor(features_df.values, dtype=torch.float32)
    num_nodes = features.shape[0]
    
    if max_nodes is None or num_nodes <= max_nodes:
        features = torch.tensor(features_df.values, dtype=torch.float32)
    else:
        padding = torch.zeros((max_nodes - num_nodes, features_df.shape[1]), dtype=torch.float32)
        features = torch.cat((torch.tensor(features_df.values, dtype=torch.float32), padding), dim=0)

    edges_df = pd.read_csv(edges_file, header=0, low_memory=False)
    edges_df = edges_df[['src', 'dst']].dropna()

    ip_to_id = {ip: idx for idx, ip in enumerate(set(edges_df['src']).union(set(edges_df['dst'])))}
    edges_df['src'] = edges_df['src'].map(ip_to_id)
    edges_df['dst'] = edges_df['dst'].map(ip_to_id)
    edges_df = edges_df[(edges_df['src'].notna()) & (edges_df['dst'].notna())]
    edges_df = edges_df[(edges_df['src'] < num_nodes) & (edges_df['dst'] < num_nodes)]

    edge_index = torch.tensor(edges_df[['src', 'dst']].values.T, dtype=torch.long)

    print(f'Number of nodes: {num_nodes}')
    print(f'Number of edges: {edges_df.shape[0]}')
    print(f'Features shape: {features.shape}')
    print(f'Edge index shape: {edge_index.shape}')

    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    graph = dgl.add_self_loop(graph)
    
    graph.ndata['feat'] = features

    return graph

def create_dataset(features_file, labels_file, edges_file):
    features_df = pd.read_csv(features_file)
    labels_df = pd.read_csv(labels_file)
    edges_df = pd.read_csv(edges_file)

    # Ensure features and graph nodes match
    num_nodes = features_df.shape[0]
    ip_to_id = {ip: idx for idx, ip in enumerate(set(edges_df['src']).union(set(edges_df['dst'])))}
    edges_df['src'] = edges_df['src'].map(ip_to_id)
    edges_df['dst'] = edges_df['dst'].map(ip_to_id)

    # Ensure edge indices are within node range
    edges_df = edges_df[(edges_df['src'].notna()) & (edges_df['dst'].notna())]
    edges_df = edges_df[(edges_df['src'] < num_nodes) & (edges_df['dst'] < num_nodes)]

    edge_list = torch.tensor(edges_df[['src', 'dst']].values.T, dtype=torch.long)
    graph = dgl.graph((edge_list[0], edge_list[1]), num_nodes=num_nodes)
    graph = dgl.add_self_loop(graph)

    features = torch.tensor(features_df.values, dtype=torch.float32)
    labels = torch.tensor(labels_df.values.flatten(), dtype=torch.long)

    return graph, features, labels

def train_model(train_features_file, train_labels_file, train_edges_file, val_features_file, val_labels_file, val_edges_file, test_features_file, test_labels_file, test_edges_file, num_nodes_per_batch, num_epochs, batch_size, lr):
    device = torch.device('cpu')  # Use CPU

    # Load the graph and features
    print("Loading graph and features...")
    train_graph = create_graph(train_features_file, train_edges_file)
    val_graph = create_graph(val_features_file, val_edges_file)
    test_graph = create_graph(test_features_file, test_edges_file)

    # Load the features and labels
    train_features = torch.tensor(pd.read_csv(train_features_file).values, dtype=torch.float32)
    train_labels = torch.tensor(pd.read_csv(train_labels_file).values.flatten(), dtype=torch.long)
    val_features = torch.tensor(pd.read_csv(val_features_file).values, dtype=torch.float32)
    val_labels = torch.tensor(pd.read_csv(val_labels_file).values.flatten(), dtype=torch.long)
    test_features = torch.tensor(pd.read_csv(test_features_file).values, dtype=torch.float32)
    test_labels = torch.tensor(pd.read_csv(test_labels_file).values.flatten(), dtype=torch.long)

    # Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_dataset = GraphDataset(train_graph, train_features, train_labels, num_nodes_per_batch)
    val_dataset = GraphDataset(val_graph, val_features, val_labels, num_nodes_per_batch)
    test_dataset = GraphDataset(test_graph, test_features, test_labels, num_nodes_per_batch)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Initialize the model
    print("Initializing the model...")
    model = GraphSAGEGATModel(
        in_feats=36,
        hidden_dim=64,
        num_classes=len(torch.unique(train_labels)),
        num_heads=4
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        print(f'Training..Epoch {epoch + 1}')
        model.train()
        epoch_loss = 0.0
        for batch_graphs, batch_features, batch_labels in train_loader:
            batch_graphs = batch_graphs.to(device)
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_graphs, batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}')

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_graphs, batch_features, batch_labels in val_loader:
                batch_graphs = batch_graphs.to(device)
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_graphs, batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        patience=3
        # Save the model checkpoint if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model checkpoint saved for epoch {epoch + 1}.')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Clear memory
        del batch_graphs, batch_features, batch_labels, outputs
        gc.collect()
        torch.cuda.empty_cache()

    print("Training complete.")
if __name__ == "__main__":
    features_file = 'data/features.csv'
    labels_file = 'data/labels.csv'
    edges_file = 'data/edges.csv'
    train_features_file='data/train_features.csv'
    train_labels_file='data/train_labels.csv'
    train_edges_file='data/train_edges.csv'
    val_features_file='data/val_features.csv'
    val_labels_file='data/val_labels.csv'
    val_edges_file='data/val_edges.csv'
    test_features_file='data/test_features.csv'
    test_labels_file='data/test_labels.csv'
    test_edges_file='data/test_edges.csv'
    split_and_save_data(features_file, labels_file, edges_file, train_features_file, val_features_file, test_features_file, train_labels_file, val_labels_file, test_labels_file, train_edges_file, val_edges_file, test_edges_file)
    train_model(
        train_features_file, train_labels_file, train_edges_file, 
        val_features_file, val_labels_file, val_edges_file, 
        test_features_file, test_labels_file, test_edges_file, 
        num_nodes_per_batch=3000, num_epochs=10, batch_size=64, lr=0.001
    )

