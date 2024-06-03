import numpy as np
import pandas as pd
from typing import List, Set

def precision_at_k(recommended_items: List[str], actual_items: Set[str], k: int) -> float:
    """Calculate top-k precision for a single list of recommendations."""
    if not recommended_items:
        return 0
    recommended_k = recommended_items[:k]
    true_positives = len(set(recommended_k) & actual_items)
    return true_positives / len(recommended_k)

def recall_at_k(recommended_items: List[str], actual_items: Set[str], k: int) -> float:
    """Calculate top-k recall for a single list of recommendations."""
    if not actual_items:
        return 0
    recommended_k = recommended_items[:k]
    true_positives = len(set(recommended_k) & actual_items)
    return true_positives / len(actual_items)

def f1_score_at_k(recommended_items: List[str], actual_items: Set[str], k: int) -> float:
    """Calculate F1 score at k."""
    precision = precision_at_k(recommended_items, actual_items, k)
    recall = recall_at_k(recommended_items, actual_items, k)
    if precision == 0 and recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def get_top_k_recommendations(data: pd.DataFrame, k: int) -> pd.Series:
    """Get top-k recommended books for each user based on predicted ratings."""
    top_k_recommendations = data.sort_values(['user_id', 'predicted_rating'], ascending=[True, False]) \
                                .groupby('user_id')['book_id'] \
                                .apply(lambda x: x.head(k).tolist())
    return top_k_recommendations

def get_actual_items(data: pd.DataFrame, threshold: int) -> pd.Series:
    """Get items that have been rated equal to or above a certain threshold."""
    relevant_items = data[data['rating'] >= threshold] \
                     .groupby('user_id')['book_id'] \
                     .apply(list)
    return relevant_items

def evaluate_recommendations(top_k_recommendations: pd.Series, actual_items: pd.Series, k: int) -> tuple:
    precision_scores = []
    recall_scores = []
    f1_scores = []

    users = set(top_k_recommendations.index).intersection(set(actual_items.index))

    for user in users:
        recommended_items = top_k_recommendations.loc[user]
        actual_items_user = set(actual_items.loc[user])
        precision = precision_at_k(recommended_items, actual_items_user, k)
        recall = recall_at_k(recommended_items, actual_items_user, k)
        f1 = f1_score_at_k(recommended_items, actual_items_user, k)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    # Averaging the scores
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)

    return mean_precision, mean_recall, mean_f1
