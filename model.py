import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv, GATConv

class GraphSAGEGATModel(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, num_heads):
        super(GraphSAGEGATModel, self).__init__()
        self.sage = SAGEConv(in_feats, hidden_dim, aggregator_type='mean')
        self.gat = GATConv(hidden_dim, hidden_dim, num_heads=num_heads)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, g, features):
        x = self.sage(g, features)
        x = F.relu(x)
        
        x = self.gat(g, x)
        x = F.relu(x)
        
        x = x.mean(dim=1)
        
        x = self.fc(x)
        
        return x
