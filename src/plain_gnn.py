import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, to_hetero
import torch.optim as optim
from torch.nn.functional import mse_loss

def load_data_from_csv(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Extract user IDs, book IDs, and ratings
    user_ids = df['user_id'].values
    book_ids = df['book_id'].values
    ratings = df['rating'].values
    
    # Convert to tensor indices
    user_tensor = torch.tensor(user_ids, dtype=torch.long)
    book_tensor = torch.tensor(book_ids, dtype=torch.long)
    rating_tensor = torch.tensor(ratings, dtype=torch.float)
    
    # Construct the edge index from user nodes to book nodes
    edge_index = torch.stack([user_tensor, book_tensor], dim=0)
    
    # Prepare data object
    data = Data(edge_index=edge_index, y=rating_tensor)
    
    # Since edge_index contains indices directly, make sure num_nodes is correctly set
    data.num_nodes = max(user_tensor.max(), book_tensor.max()) + 1
    
    return data, user_tensor, book_tensor


class GraphSAGEModel(torch.nn.Module):
    def __init__(self, num_nodes, hidden_dim, metadata):
        super(GraphSAGEModel, self).__init__()
        self.node_features = torch.nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.conv1 = SAGEConv(-1, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        # self.model = to_hetero(torch.nn.Sequential(self.conv1, self.conv2), metadata, aggr='sum')
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x, edge_index):
        x = self.node_features
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def train_model(self, data, user_tensor, book_tensor, epochs=200):
        for epoch in range(epochs):
            
            self.train()
            self.optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            # Calculate loss between predicted and actual ratings
            predicted_ratings = (out[user_tensor] * out[book_tensor]).sum(dim=1)
            loss = mse_loss(predicted_ratings, data.y)
            loss.backward()
            self.optimizer.step()
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Example usage
# Assuming 'data', 'user_tensor', 'book_tensor' are defined as shown previously



if __name__ == "__main__":
    data, user_tensor, book_tensor = load_data_from_csv('data/ratings.csv')
    metadata = {'num_node_types': 2, 'num_edge_types': 1}  # Update this based on your specific metadata needs
    model = GraphSAGEModel(num_nodes=len(user_tensor) + len(book_tensor), hidden_dim=32, metadata=metadata)
    # print(model(data.x, data.edge_index))

    model.train_model(data, user_tensor, book_tensor)


