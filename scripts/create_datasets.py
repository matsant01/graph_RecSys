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
import numpy as np
import argparse
import os


# make result reproducible
SEED = 42


class LoadData:
    def __init__(self, book_path, ratings_path, device, user_to_sample=None):
        self.df_books = pd.read_csv(book_path)[['book_id', 'title', 'authors', 'language_code']]   
        self.df_ratings = pd.read_csv(ratings_path)
        
        seed_everything(42)  
        random.seed(42)  

        # sample a subset of users and all their ratings
        if user_to_sample:
            sampled_users = np.random.choice(self.df_ratings.user_id.unique(), user_to_sample, replace=False)
            self.df_ratings = self.df_ratings[self.df_ratings.user_id.isin(sampled_users)]

        self.device = device


    def compute_books_embeddings(self, df_books, include_authors=True):
        """
        Computes the sentence embeddings for the book titles and authors, to be used as node features in the graph.
        :param df_books: DataFrame with the books data
        :param include_authors: Boolean to include authors in the embeddings (to set to false if authors are separate nodes of the graph)
        :return: torch.tensor with the embeddings
        """
        
        model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        df_books["text_to_embed"] = "Title: " + df_books["title"]
        if include_authors:
            df_books["text_to_embed"] = df_books["text_to_embed"] + " Authors: " + df_books["authors"]
        with torch.no_grad():
            titles_emb = model.encode(df_books['text_to_embed'].values, device=self.device, show_progress_bar=True, batch_size=128)
            
        del model
        torch.cuda.empty_cache()    

        books_features = torch.tensor(titles_emb)
        return books_features
    
    
    def compute_authors_embeddings(self, df_books):
        """
        Computes the sentence embeddings for the authors to be used as node features in the graph.
        :param df_books: DataFrame with the books data
        :return: torch.tensor with the embeddings
        """
        
        model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        with torch.no_grad():
            authors_emb = model.encode(df_books['single_authors'].unique(), device=self.device, show_progress_bar=True, batch_size=128)
            
        del model
        torch.cuda.empty_cache()    

        authors_features = torch.tensor(authors_emb)
        return authors_features
    
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
    
    def encode_string_array(self, array):
        unique_strings = array.unique()
        string_to_int = {string: i for i, string in enumerate(unique_strings)}
        int_array = np.vectorize(string_to_int.get)(array)
        return int_array

    def build_edge_index(self, column1, column2):

        edge_index = torch.stack([
            torch.tensor(column1), 
            torch.tensor(column2)]
            , dim=0)
        return edge_index


    def create_hetero_graph(self, books_features, users_features, add_data): 

        ratings_book_merged = pd.merge(self.df_ratings, self.df_books, on='book_id')

        # user and book ids aren't continguos, we need to map them to a contiguous range
        ratings_book_merged = self.map_user_book_ids(ratings_book_merged)

        data = HeteroData()
        data['user'].x = torch.tensor(users_features).detach().to(torch.float32)  # [num_users]
        data['book'].x = books_features.clone().detach().to(torch.float32) # (num_books, num_books_features)

        data['user', 'rates', 'book'].edge_index = self.build_edge_index(ratings_book_merged['mapped_user_id'], ratings_book_merged['mapped_book_id'])
        rating = torch.from_numpy(ratings_book_merged['rating'].values).float()
        data['user', 'rates', 'book'].edge_label = rating  # [num_ratings]

        if add_data:
            # Split the 'authors' column into lists
            ratings_book_merged['single_authors'] = ratings_book_merged['authors'].str.split(', ').copy()
            expanded_df = ratings_book_merged.explode('single_authors', ignore_index=True)
            int_array = self.encode_string_array(expanded_df['single_authors'])
            data['book', 'by', 'author'].edge_index = self.build_edge_index(expanded_df['mapped_book_id'].values, int_array)
            data['author'].x = self.compute_authors_embeddings(expanded_df)
            assert data['author'].x.shape[0] == max(data['book', 'by', 'author'].edge_index[1] + 1)

            int_array = self.encode_string_array(ratings_book_merged['language_code'])
            data['book', 'written_in', 'language'].edge_index = self.build_edge_index(ratings_book_merged['mapped_book_id'].values, int_array)
            data['language'].x = torch.nn.functional.one_hot(torch.tensor(np.unique(int_array)), num_classes=len(np.unique(int_array)))

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
        seed_everything(42)  
        random.seed(42) 

        tfs = RandomLinkSplit(is_undirected=True, 
                            num_val=0.075,
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
    parser = argparse.ArgumentParser(description='Process and save hetero graph data.')
    parser.add_argument('--save_dir', type=str, help='Directory where the data will be saved')
    parser.add_argument('--add_extra_data', action='store_true', help='Add extra data to the graph (authors, languages, etc.)')

    args = parser.parse_args()
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    book_path = 'data/GoodBooks-10k/books.csv'
    ratings_path = 'data/GoodBooks-10k/ratings.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader = LoadData(book_path, ratings_path, device)
    books_features = loader.compute_books_embeddings(loader.df_books, include_authors=False)
    users_features = loader.compute_user_embeddings(loader.df_ratings)
    print('adding extra data:', args.add_extra_data)    
    data = loader.create_hetero_graph(books_features, users_features, args.add_extra_data)

    print(data)
    train_data, val_data, test_data = loader.split_hetero(data) 
    print(train_data)
    print(val_data)
    print(test_data)

    # saving both as torch and csv
    loader.save_hetero(data, f'{save_dir}/data_hetero.pt')
    loader.save_hetero(train_data, f'{save_dir}/train_hetero.pt')
    loader.save_hetero(val_data, f'{save_dir}/val_hetero.pt')
    loader.save_hetero(test_data, f'{save_dir}/test_hetero.pt')
    loader.convert_hetero_to_csv_save(train_data, f'{save_dir}/train.csv')
    loader.convert_hetero_to_csv_save(val_data, f'{save_dir}/val.csv')
    loader.convert_hetero_to_csv_save(test_data, f'{save_dir}/test.csv')



