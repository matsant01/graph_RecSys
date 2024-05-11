import numpy as np
import pandas as pd

def initialize_data(filepath: str, num_factors: int = 5):
    """
    Loads the dataset from a CSV file and initializes user and book factor matrices.

    Args:
        filepath (str): Path to the CSV file containing the data.
        num_factors (int): Number of latent factors to use in the matrices.

    Returns:
        tuple: A tuple containing the dataset (DataFrame), user factors matrix (W),
               book factors matrix (Z), number of users, and number of books.
    """
    data = pd.read_csv(filepath)
    num_users = len(data['user_id'].unique())
    num_books = len(data['book_id'].unique())
    
    W = np.random.normal(scale=1./num_factors, size=(num_users, num_factors))
    Z = np.random.normal(scale=1./num_factors, size=(num_books, num_factors))
    
    return data, W, Z, num_users, num_books


def factorize_matrix_SGD(data: pd.DataFrame, W: np.ndarray, Z: np.ndarray, num_iterations: int = 10000, batch_size: int = 32, learning_rate: float = 0.01):
    """
    Performs stochastic gradient descent to optimize the factor matrices W and Z.

    Args:
        data (pd.DataFrame): Dataset containing user IDs, book IDs, and ratings.
        W (np.ndarray): User factors matrix.
        Z (np.ndarray): Book factors matrix.
        num_iterations (int): Number of iterations to run SGD.
        batch_size (int): Number of samples in each batch.
        learning_rate (float): Learning rate for updates.

    Returns:
        tuple: Updated user factors matrix (W) and book factors matrix (Z).
    """
    for _ in range(num_iterations):
        batch = data.sample(n=batch_size, replace=True)
        users = batch['user_id'].values - 1
        books = batch['book_id'].values - 1
        ratings = batch['rating'].values

        predictions = np.sum(W[users, :] * Z[books, :], axis=1)
        errors = ratings - predictions

        for idx in range(batch_size):
            u = users[idx]
            b = books[idx]
            W[u, :] += learning_rate * errors[idx] * Z[b, :]
            Z[b, :] += learning_rate * errors[idx] * W[u, :]
    return W, Z


def compute_predictions(data: pd.DataFrame, W: np.ndarray, Z: np.ndarray) -> pd.DataFrame:
    """
    Computes the predicted ratings for each user and book pair in the data.

    Args:
        data (pd.DataFrame): Dataset containing user IDs and book IDs.
        W (np.ndarray): User factors matrix.
        Z (np.ndarray): Book factors matrix.

    Returns:
        pd.DataFrame: Updated DataFrame including a new column with predicted ratings.
    """
    data['predicted_rating'] = data.apply(
        lambda row: np.dot(W[row['user_id']-1, :], Z[row['book_id']-1, :].T), axis=1)
    return data


def calculate_mse(data: pd.DataFrame) -> float:
    """
    Calculates the mean squared error (MSE) between actual and predicted ratings.

    Args:
        data (pd.DataFrame): DataFrame containing actual and predicted ratings.

    Returns:
        float: The calculated mean squared error.
    """
    mse = np.mean((data['rating'] - data['predicted_rating']) ** 2)
    return mse
