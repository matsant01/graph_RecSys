import os
import torch
import torch_geometric as pyg
from torch_geometric.data import HeteroData

os.environ['TORCH'] = torch.__version__

from torch_geometric.nn import SAGEConv, to_hetero
from torch import Tensor

class SAGEConv2Encoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        # Takes the edge_index (not the edge_label_index) as input, and performs
        # message passing on the graph.
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EncDec_v3(torch.nn.Module):
    def __init__(self, hidden_channels: int, data: HeteroData, book_channels: int = 384):
        super().__init__()
        
        self.book_lin = torch.nn.Linear(book_channels, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.book_emb = torch.nn.Embedding(data["book"].num_nodes, hidden_channels)
        
        self.encoder = SAGEConv2Encoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        # self.decoder =    using only the dot-product for decoding

    def forward(self, data: HeteroData):
        """
        Forward pass of the model. The input data is a HeteroData object, which
        contains the following:
        - data["user"].n_id: the node indices of the user nodes
        - data["book"].n_id: the node indices of the book nodes
        - data["book"].x: the feature matrix of the book nodes
        - data.edge_index_dict: a dictionary containing the edge indices of all edge types (for convolution)
        - data["user", "rates", "book"].edge_label_index: the edge indices for supervision
        
        """
        
        # This is completely ignoring the user features and only caring about the book features.
        x_dict = {
          "user": self.user_emb(data["user"].n_id),
          "book": self.book_lin(data["book"].x) + self.book_emb(data["book"].n_id),
        }
                
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.encoder(x_dict, data.edge_index_dict)
        
        # Applying the dot product to compute the predictions
        row, col = data["user", "rates", "book"].edge_label_index
        return (x_dict["user"][row] * x_dict["book"][col]).sum(dim=-1)