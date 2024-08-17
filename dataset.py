import torch
import dgl
from torch.utils.data import Dataset

def sample_subgraph(graph, num_nodes_per_batch):
    """Sample a subgraph from the original graph."""
    nodes = torch.randperm(graph.num_nodes())[:num_nodes_per_batch]
    subgraph = graph.subgraph(nodes)
    return subgraph

class GraphDataset(Dataset):
    def __init__(self, graph, features, labels, num_nodes_per_batch):
        self.graph = graph
        self.features = features
        self.labels = labels
        self.num_nodes_per_batch = num_nodes_per_batch

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        subgraph = sample_subgraph(self.graph, self.num_nodes_per_batch)
        subgraph_features = self.features[subgraph.ndata[dgl.NID]]
        subgraph_labels = self.labels[subgraph.ndata[dgl.NID]]
        return subgraph, subgraph_features, subgraph_labels

def collate_fn(batch):
    graphs, features, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.cat(features), torch.cat(labels)
