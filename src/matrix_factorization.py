import numpy as np
import pandas as pd
from tqdm import tqdm as tp


class MatrixFactorization:
    def __init__(self, num_factors: int = 5, num_epochs: int = 1, batch_size: int = 1024, learning_rate: float = 0.01, lambda_reg: float = 0.01, log_every: int = 10000):
        self.num_factors = num_factors
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.log_every = log_every

    def train_loop(self, full_data, train_data):
        self.initialize_weights(full_data)
        self.factorize_matrix_SGD(train_data)
        return self

    def predict(self, user_book_pairs) -> pd.DataFrame:
        """Make prediction for every pair of user and book id"""
        user_id = user_book_pairs["user_id"].values
        book_id = user_book_pairs["book_id"].values

        predictions = np.sum(self.user_factors_[user_id, :] * self.book_factors_[book_id, :], axis=1)
        return predictions

    def evaluate(self, data: pd.DataFrame):
        predictions = self.predict(data)
        mse = self.calculate_mse(data["rating"], predictions)
        return mse

    def initialize_weights(self, full_data:pd.DataFrame):
        
        # matrix needs to be created on the full data otherwise indices won't match on the training set
        self.num_users = len(full_data["user_id"].unique())
        self.num_books = len(full_data["book_id"].unique())

        # Initialize user and book factors
        self.user_factors_ = np.random.rand(self.num_users, self.num_factors)
        self.book_factors_ = np.random.rand(self.num_books, self.num_factors)


    def factorize_matrix_SGD(self, data):

        cumulative_loss = []
        for epoch in range(self.num_epochs):
            # Shuffle the data for each epoch
            data = data.sample(frac=1)
            done_batches  = 0 

            # Iterate over data in batches
            for i in tp(range(0, len(data), self.batch_size)):  # TODO: replace with proper data loader? 
                batch = data[i:i + self.batch_size]

                # get index of users and books in the batch and labels
                users = batch["user_id"].values 
                books = batch["book_id"].values 
                labels = batch["rating"].values

                predictions = self.predict(batch)
                errors = labels - predictions

                # Update factors based on the batch errors
                for idx in range(len(batch)):
                    u = users[idx]
                    b = books[idx]
                    W_grad = -errors[idx] * self.book_factors_[b, :] + self.lambda_reg * self.user_factors_[u, :]
                    Z_grad = -errors[idx] * self.user_factors_[u, :] + self.lambda_reg * self.book_factors_[b, :]
                    self.user_factors_[u, :] -= self.learning_rate * W_grad
                    self.book_factors_[b, :] -= self.learning_rate * Z_grad

                # logging
                # cumulative_loss.append(self.calculate_mse(labels, predictions))

                # if done_batches % self.log_every == 0:
                #     print(f"Epoch: {epoch}, MSE: {np.mean(cumulative_loss)}")
                #     cumulative_loss = []



    def calculate_mse(self, labels, predictions) -> float:
        """
        Calculates the mean squared error (MSE) between actual and predicted ratings.

        Args:
            data (pd.DataFrame): DataFrame containing actual and predicted ratings.

        Returns:
            float: The calculated mean squared error.
        """
        mse = np.mean((labels - predictions) ** 2)
        return mse



if __name__ == "__main__":
    full_datadata = pd.read_csv("data/ratings.csv")
    train_data = pd.read_csv("data/splitted_data/train.csv")
    test_data = pd.read_csv("data/splitted_data/test.csv")
    mf = MatrixFactorization()
    mf.train_loop(full_datadata, train_data)
    mse = mf.evaluate(test_data)

