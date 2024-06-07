import numpy as np
import pandas as pd
from typing import List, Set

def precision_at_k(user_ratings: pd.DataFrame, threshold:int, k: int) -> float:
    """Calculate top-k precision for a single list of recommendations."""

    relevant_items = user_ratings[(user_ratings['rating'] >= threshold)].sort_values('rating', ascending=False).head(k)
    relevant_recommended_items = len(relevant_items[relevant_items['predicted_rating'] >= threshold])

    # if there are no relevant items, precision is 1 as there is nothing to recommend
    if len(relevant_items) == 0:
        return 1
    return relevant_recommended_items / len(relevant_items)

def recall_at_k(user_ratings: pd.DataFrame, threshold:int, k: int) -> float:
    """Calculate top-k recall for a single list of recommendations."""
    relevant_items = user_ratings[(user_ratings['rating'] >= threshold)].sort_values('rating', ascending=False)
    tot_relevant_items = len(relevant_items)
    relevant_items = relevant_items.head(k)
    relevant_recommended_items = len(relevant_items[relevant_items['predicted_rating'] >= threshold])
    
    # if there are no relevant items, precision is 1 as there is nothing to recommend
    if relevant_items.shape[0] == 0:
        return 1
    
    return relevant_recommended_items / tot_relevant_items

def f1_score_at_k(precision, recall) -> float:
    """Calculate F1 score at k."""
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_recommendations(data: pd.DataFrame, threshold:int, k: int, n_precisions) -> tuple:
    precision_scores = []
    recall_scores = []
    f1_scores = []
    map_k = []

    for user in data.user_id.unique():
        user_ratings = data[data['user_id'] == user]
        precision = precision_at_k(user_ratings, threshold, k)
        recall = recall_at_k(user_ratings, threshold, k)
        f1 = f1_score_at_k(precision, recall)
        precisions = [precision_at_k(user_ratings, threshold, i) for i in range(1, n_precisions)]

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        map_k.append(np.mean(precisions))

    # Averaging the scores
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    map_k = np.mean(map_k)

    return mean_precision, mean_recall, mean_f1, map_k
