from torch_geometric import seed_everything
import random
import pandas as pd
import torch
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from torch_geometric.transforms import RandomLinkSplit

# make result reproducible
seed_everything(42)  
random.seed(42)  


class LoadData:
    def __init__(self, book_path, ratings_path, device, sample_size=1.0):
        self.df_books = pd.read_csv(book_path)[['book_id', 'title', 'authors']]   
        self.df_ratings = pd.read_csv(ratings_path).sample(frac=sample_size, random_state=42)
        self.device = device


    def compute_books_embeddings(self,df_books):

        model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        df_books["text_to_embed"] = "Title: " + df_books["title"] + " Authors: " + df_books["authors"]
        with torch.no_grad():
            titles_emb = model.encode(df_books['text_to_embed'].values, device=self.device, show_progress_bar=True, batch_size=32)
            
        del model
        torch.cuda.empty_cache()    

        books_features = torch.tensor(titles_emb)
        # print("Books features shape:", books_features.shape)
        return books_features
    
    def compute_user_embeddings(self,df_ratings):
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
        users_features = features_scaled.to_numpy()

        return users_features

    
    def map_user_book_ids(self, ratings_book_merged):
        """Makses user and book ids contiguous.

        Args:
            ratings_book_merged (_type_): _description_

        Returns:
            _type_: _description_
        """
        unique_user_id = ratings_book_merged['user_id'].unique()
        unique_user_id = pd.DataFrame(data={
            'user_id': unique_user_id, 
            'mapped_user_id': pd.RangeIndex(len(unique_user_id))
            })
        
        unique_book_id = ratings_book_merged['book_id'].unique()
        unique_book_id = pd.DataFrame(data={
            'book_id': unique_book_id,
            'mapped_book_id': pd.RangeIndex(len(unique_book_id))
            })

        ratings_book_merged = ratings_book_merged.merge(unique_user_id, on='user_id')
        ratings_book_merged = ratings_book_merged.merge(unique_book_id, on='book_id')

        return ratings_book_merged


    def create_hetero_graph(self, books_features, users_features): 
        ratings_book_merged = pd.merge(self.df_ratings, self.df_books, on='book_id')

        # user and book ids aren't continguos, we need to map them to a contiguous range
        ratings_book_merged = self.map_user_book_ids(ratings_book_merged)

        # With this, we are ready to create the edge_index representation in COO format
        # following the PyTorch Geometric semantics:
        edge_index = torch.stack([
            torch.tensor(ratings_book_merged['mapped_user_id'].values), 
            torch.tensor(ratings_book_merged['mapped_book_id'].values)]
            , dim=0)
    
        data = HeteroData()
        data['user'].x = torch.tensor(users_features,).detach().float() # (num_users, num_users_features)
        data['book'].x = torch.tensor(books_features,).detach().float() # (num_books, num_books_features)

        # Add the rating edges:
        data['user', 'rates', 'book'].edge_index = edge_index  # (2, num_ratings)

        # # Add the rating labels:
        rating = torch.from_numpy(ratings_book_merged['rating'].values).float()
        data['user', 'rates', 'book'].edge_label = rating  # [num_ratings]

        # We also need to make sure to add the reverse edges from books to users
        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        data = T.ToUndirected()(data)

        # With the above transformation we also got reversed labels for the edges. We remove them
        del data['book', 'rev_rates', 'user'].edge_label

        # print(data['user'].num_nodes,len(ratings_book_merged['user_id'].unique()))
        assert data['user'].num_nodes == len(ratings_book_merged['user_id'].unique())
        assert data['user', 'rates', 'book'].num_edges == len(ratings_book_merged)

        return data
    
    def split_hetero(self, data):
        ## designed for transductive learning
        tfs = RandomLinkSplit(is_undirected=True, 
                            num_val=0.1,
                            num_test=0.1,
                            neg_sampling_ratio=0.0,
                            add_negative_train_samples=False,
                            edge_types=[('user', 'rates', 'book')],
                            rev_edge_types=[('book', 'rev_rates', 'user')],
                            )

        train_data, val_data, test_data = tfs(data)
        return train_data, val_data, test_data
    
    def save_hetero(self, data, path):
        torch.save(data, path)

    def load_hetero(self, path):
        data = torch.load(path)
        return data

    def convert_hetero_to_csv_save(self, data, path):
        labels = data['user', 'rates', 'book'].edge_label
        label_index =  data['user', 'rates', 'book'].edge_label_index

        concat = torch.cat([label_index, labels.unsqueeze(0)], dim=0)
        final_dataset = pd.DataFrame(concat.T.numpy(), columns=["user_id", "book_id", "rating"])
        final_dataset['user_id'] = final_dataset['user_id'].astype(int)
        final_dataset['book_id'] = final_dataset['book_id'].astype(int)
        final_dataset.to_csv(path, index=False)



if __name__ == "__main__":
    book_path = 'data/books.csv'
    ratings_path = 'data/ratings.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader = LoadData(book_path, ratings_path, device)
    books_features = loader.compute_books_embeddings(loader.df_books)
    users_features = loader.compute_user_embeddings(loader.df_ratings)
    data = loader.create_hetero_graph(books_features, users_features)

    train_data, val_data, test_data = loader.split_hetero(data) 

    # saving both as torch and csv
    loader.save_hetero(data, 'data/splitted_data/data_hetero.pt')
    loader.save_hetero(train_data, 'data/splitted_data/train_hetero.pt')
    loader.save_hetero(val_data, 'data/splitted_data/val_hetero.pt')
    loader.save_hetero(test_data, 'data/splitted_data/test_hetero.pt')
    loader.convert_hetero_to_csv_save(train_data, 'data/splitted_data/train.csv')
    loader.convert_hetero_to_csv_save(val_data, 'data/splitted_data/val.csv')
    loader.convert_hetero_to_csv_save(test_data, 'data/splitted_data/test.csv')



