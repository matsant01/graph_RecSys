import os
import sys
from tqdm import tqdm
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.loader import LinkNeighborLoader
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from sentence_transformers import SentenceTransformer

sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.models import EncDec_v3

def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print("Using device: ", device)
    
    
    ############# Load the data #############
    df_books = pd.read_csv('../data/books.csv')[['book_id', 'title', 'authors']]
    df_ratings = pd.read_csv('../data/ratings.csv')
    len("{} ratings loaded".format(df_ratings))

    ############# Create Books Node Features #############
    df_books["text_to_embed"] = "Title: " + df_books["title"] + " Authors: " + df_books["authors"]
    with torch.no_grad():
        titles_emb = model.encode(df_books['text_to_embed'].values, device=device, show_progress_bar=True, batch_size=32)
    books_features = torch.tensor(titles_emb)
        
    del model
    torch.cuda.empty_cache()    

    ############# Create Heterogeneous Graph #############
    # embedding users

    # Create a bipartite graph
    B = nx.Graph()
    # Add nodes with the node attribute "bipartite"
    B.add_nodes_from(df_ratings['user_id'].unique(), bipartite=0)  # Users
    B.add_nodes_from(df_ratings['book_id'].unique(), bipartite=1)  # Books

    # Add edges between users and books
    for _, row in tqdm(df_ratings.iterrows(), total=df_ratings.shape[0], desc="Adding edges"):
        B.add_edge(row['user_id'], row['book_id'], weight=row['rating'])

    # Compute metrics
    centrality = nx.degree_centrality(B)
    print('degree centrality computed')
    pagerank = nx.pagerank(B, weight='weight')
    print('pagerank computed')
    average_rating = df_ratings.groupby('user_id')['rating'].mean()
    print('all metrics computed')

    # # Prepare feature vectors for users
    features = pd.DataFrame(index=df_ratings['user_id'].unique())
    features['degree'] = [centrality[node] for node in features.index]
    features['pagerank'] = [pagerank[node] for node in features.index]
    features['average_rating'] = [average_rating.get(node, 0) for node in features.index]  # Add average ratings

    # # Normalize features
    scaler = MinMaxScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features), index=features.index, columns=features.columns)

    # # Display the normalized features
    users_features = features_scaled.to_numpy(dtype=np.float32)

    # Merge the two dataframes keeping user_id, book_id, rating, title, authors
    df_ratings = pd.merge(df_ratings, df_books, on='book_id')

    # Create a mapping from the user_id to a unique consecutive value in the range [0, num_users]:
    unique_user_id = df_ratings['user_id'].unique()
    unique_user_id = pd.DataFrame(data={
        'user_id': unique_user_id, 
        'mapped_user_id': pd.RangeIndex(len(unique_user_id))
        })
    print("Mapping of user IDs to consecutive values:")
    print("==========================================")
    print(unique_user_id.head())
    print()

    # Create a mapping from the book_id to a unique consecutive value in the range [0, num_books]:
    unique_book_id = df_ratings['book_id'].unique()
    unique_book_id = pd.DataFrame(data={
        'book_id': unique_book_id,
        'mapped_book_id': pd.RangeIndex(len(unique_book_id))
        })
    print("Mapping of book IDs to consecutive values:")
    print("===========================================")
    print(unique_book_id.head())
    print()

    df_ratings = df_ratings.merge(unique_user_id, on='user_id')
    df_ratings = df_ratings.merge(unique_book_id, on='book_id')

    # With this, we are ready to create the edge_index representation in COO format
    # following the PyTorch Geometric semantics:
    edge_index = torch.stack([
        torch.tensor(df_ratings['mapped_user_id'].values), 
        torch.tensor(df_ratings['mapped_book_id'].values)]
        , dim=0)

    print(edge_index[:, :10])
    
    # Create the heterogeneous graph data object:
    data = HeteroData()

    data['user'].x = torch.tensor(users_features,)  # (num_users, num_users_features)
    data['book'].x = torch.tensor(titles_emb,)  # (num_books, num_books_features)

    data['user', 'rates', 'book'].edge_index = edge_index  # (2, num_ratings)
    rating = torch.from_numpy(df_ratings['rating'].values)
    data['user', 'rates', 'book'].edge_label = rating  # [num_ratings]

    data = T.ToUndirected()(data)
    del data['book', 'rev_rates', 'user'].edge_label

    print(data['user'].num_nodes,len(unique_user_id))
    assert data['user'].num_nodes == len(unique_user_id)
    assert data['user', 'rates', 'book'].num_edges == len(df_ratings)


    ################# Create Data Splits #################
    train_data, val_data, test_data = T.RandomLinkSplit(
        add_negative_train_samples=False,
        num_val=0.15,
        num_test=0.15,
        edge_types=[('user', 'rates', 'book')],
        rev_edge_types=[('book', 'rev_rates', 'user')],
    )(data)
    print("Slitted data")
    print("Train data:", train_data)
    print("Validation data:", val_data)
    print("Test data:", test_data)
    
    ################# Create the model #################
    model = EncDec_v3(hidden_channels=64, data=data).to(device)
    print("Model created")
    
    ################# Create the Loaders #################
    # Define seed edges:
    edge_label_index = train_data["user", "rates", "book"].edge_label_index
    edge_label = train_data["user", "rates", "book"].edge_label

    train_loader = LinkNeighborLoader(
        data=train_data,  # TODO
        num_neighbors=[50, 50],  # TODO
        neg_sampling_ratio=2,  # TODO
        edge_label_index=(("user", "rates", "book"), edge_label_index),
        edge_label=edge_label,
        batch_size=1024,
        shuffle=True,
    )
    
    val_loader = LinkNeighborLoader(
        data=val_data,  # TODO
        num_neighbors=[50, 50],  # TODO
        neg_sampling_ratio=2,  # TODO
        edge_label_index=(("user", "rates", "book"), edge_label_index),
        edge_label=edge_label,
        batch_size=1024,
        shuffle=True,
    )


    ################# Train the model #################
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logging_steps = 50
    train_losses = []
    valid_losses = []
    
    for epoch in range(1, 6):
        total_loss = total_examples = 0
        
        for i, sampled_data in enumerate(train_loader):
            optimizer.zero_grad()
            batch = sampled_data.to(device)
            pred = model.forward(batch)
            loss = F.mse_loss(pred, batch["user", "rates", "book"].edge_label.to(torch.float32))
            
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
            
            # Update progress bar
            if i % logging_steps == 0:
                est_loss = total_loss / total_examples
                print("Epoch {epoch} - Train loss {est_loss:.5f}")
                train_losses.append(est_loss)
                
        # Run Validation
        with torch.no_grad():
            model.eval()
            total_valid_loss = total_valid_examples = 0
            for batch in tqdm(val_loader, desc=f"Validation {epoch:03d}"):
                batch = batch.to(device)
                pred = model.forward(batch)
                loss = F.mse_loss(pred, batch["user", "rates", "book"].edge_label.to(torch.float32))
                
                total_valid_loss += float(loss) * pred.numel()
                total_valid_examples += pred.numel()
            valid_loss = total_valid_loss / total_valid_examples
            valid_losses.append(valid_loss)
            print(f"Validation loss: {valid_loss}")
            model.train()
            
    
    ################# Save the model #################
    torch.save(model.state_dict(), 'model_v3_{}.pt'.format(datetime.now().strftime("%Y%m%d%H%M%S")))
    
    
    ################# Plot the losses #################
    plt.plot(np.linspace(0, len(valid_losses) - 1, len(train_losses)),train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('losses.png')
    
    
    ################# Evaluate the model #################
    # test model
    test_loader = LinkNeighborLoader(
        data=test_data,  # TODO
        num_neighbors=[50, 50],  # TODO
        neg_sampling_ratio=2,  # TODO
        edge_label_index=(("user", "rates", "book"), edge_label_index),
        edge_label=edge_label,
        batch_size=1024,
        shuffle=True,
    )
    
    with torch.no_grad():
        model.eval()
        total_test_loss = total_test_examples = 0
        for batch in tqdm(test_loader, desc=f"Test"):
            batch = batch.to(device)
            pred = model.forward(batch)
            loss = F.mse_loss(pred, batch["user", "rates", "book"].edge_label.to(torch.float32))

            total_test_loss += float(loss) * pred.numel()
            total_test_examples += pred.numel()
        test_loss = total_test_loss / total_test_examples
        print(f"\n\n\nTest loss: {test_loss}")
    
    
if __name__ == '__main__':
    main()