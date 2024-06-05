import numpy as np
import pandas as pd
from tqdm import tqdm as tp
import os
from torch.utils.tensorboard import SummaryWriter

class MatrixFactorization:
    def __init__(
        self,
        num_factors: int = 5,
        num_epochs: int = 2,
        batch_size: int = 1024,
        learning_rate: float = 0.01,
        lambda_reg: float = 0.01,
        log_every: int = 100,
        output_dir: str = "./outut",
    ):
        """
        num_factors: number of latent variables
        num_epochs: number of epochs to train the model
        batch_size: size of the batch to use for training
        learning_rate: learning rate for the optimizer
        lambda_reg: regularization parameter
        log_every: frequency of batches to log the loss
        """
        self.num_factors = num_factors
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.log_every = log_every
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))


    def predict(self, user_book_pairs) -> pd.DataFrame:
        """Make prediction for every pair of user and book id"""
        user_id = user_book_pairs["user_id"].values
        book_id = user_book_pairs["book_id"].values

        predictions = np.sum(
            self.user_factors_[user_id, :] * self.book_factors_[book_id, :], axis=1
        )
        return predictions

    def evaluate(self, data: pd.DataFrame):
        predictions = self.predict(data)
        mse = self.calculate_mse(data["rating"], predictions)
        return mse

    def initialize_weights(self, full_data: pd.DataFrame):
        # matrix needs to be created on the full data otherwise indices won't match on the training set
        self.num_users = len(full_data["user_id"].unique())
        self.num_books = len(full_data["book_id"].unique())

        # Initialize user and book factors
        self.user_factors_ = np.random.rand(self.num_users, self.num_factors)
        self.book_factors_ = np.random.rand(self.num_books, self.num_factors)

    def train_loop(self, full_data, train_data, eval_data):
        self.initialize_weights(full_data)
        cumulative_loss = []

        for epoch in range(self.num_epochs):
            # Shuffle the data for each epoch
            train_data = train_data.sample(frac=1)
            done_batches = 0

            # Iterate over data in batches
            for i in tp(
                range(0, len(train_data), self.batch_size)
            ):  
                batch = train_data[i : i + self.batch_size]

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
                    W_grad = (
                        -errors[idx] * self.book_factors_[b, :]
                        + self.lambda_reg * self.user_factors_[u, :]
                    )
                    Z_grad = (
                        -errors[idx] * self.user_factors_[u, :]
                        + self.lambda_reg * self.book_factors_[b, :]
                    )
                    self.user_factors_[u, :] -= self.learning_rate * W_grad
                    self.book_factors_[b, :] -= self.learning_rate * Z_grad

                # logging
                batch_mse = self.calculate_mse(labels, predictions)
                cumulative_loss.append(batch_mse)

                if done_batches % self.log_every == 0:
                    avg_loss = np.mean(cumulative_loss)
                    self.writer.add_scalar("train/loss", avg_loss, epoch * len(train_data) // self.batch_size + done_batches)
                    # print(f"Epoch: {epoch}, Batch: {done_batches}, MSE: {avg_loss}")
                    cumulative_loss = []

                done_batches += 1

            # Evaluation step at the end of each epoch
            eval_labels = eval_data["rating"].values

            eval_predictions = self.predict(eval_data)
            eval_loss = self.calculate_mse(eval_labels, eval_predictions)
            self.writer.add_scalar("eval/loss", eval_loss, epoch)
            print(f"Epoch: {epoch}, Evaluation MSE: {eval_loss}")

        self.writer.close()
        return self

    def calculate_mse(self, labels, predictions) -> float:
        mse = np.mean((labels - predictions) ** 2)
        return mse


if __name__ == "__main__":
    full_data = pd.read_csv("data/ratings.csv")
    train_data = pd.read_csv("data/splitted_data/train.csv")
    eval_data = pd.read_csv("data/splitted_data/val.csv")
    test_data = pd.read_csv("data/splitted_data/test.csv")
    mf = MatrixFactorization()
    mf.train_loop(full_data, train_data, eval_data)
    mse = mf.evaluate(test_data)
    print(f"Test MSE: {mse}")
