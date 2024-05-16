import os
import torch
import torch_geometric as pyg


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric as pyg
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.datasets import GitHub
from torchmetrics import Metric
from torchmetrics.classification import Accuracy, BinaryF1Score, Precision, Recall

os.environ['TORCH'] = torch.__version__

class GCNRecSys(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_users: int,
        num_items: int,
        
    ): 
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, in_channels)
        self.item_embedding = nn.Embedding(num_items, in_channels)
        
        self.conv1 = pyg.nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = pyg.nn.GCNConv(hidden_channels, 1)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, edge_index):
        """
        Compute the forward pass of the model.
        First we do a convolution on the node embeddings (user and item embeddings),
        then we 
        
        """
        
        return self.sigmoid(x)