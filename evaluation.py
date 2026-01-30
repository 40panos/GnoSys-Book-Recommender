"""
Gnosys Evaluation Module
------------------------
This module provides a comprehensive suite of metrics to evaluate the performance
of recommendation models. It includes ranking metrics such as Precision@K, 
Recall@K, NDCG, and Hit Rate to assess the quality of top-K recommendations.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import logging
import utils

# --- SETUP LOGGER & CONFIG ---
logger = logging.getLogger(__name__)
config = utils.load_config()

# ==============================================================================
# 1. RANKING METRICS HELPERS
# ==============================================================================

def get_hit_rate(recommended_items, relevant_items):
    """
    Calculates the Hit Rate for a single user.
    Returns 1 if at least one relevant item is in the recommendation list, else 0.
    """
    return 1 if len(set(recommended_items) & set(relevant_items)) > 0 else 0

def get_precision_recall(recommended_items, relevant_items):
    """
    Computes Precision and Recall for a set of recommendations.
    
    Precision: Proportion of recommended items that are relevant.
    Recall: Proportion of relevant items that were successfully recommended.
    """
    intersection = len(set(recommended_items) & set(relevant_items))
    precision = intersection / len(recommended_items) if len(recommended_items) > 0 else 0
    recall = intersection / len(relevant_items) if len(relevant_items) > 0 else 0
    return precision, recall

def get_ndcg(recommended_items, relevant_items):
    """
    Computes Normalized Discounted Cumulative Gain (NDCG).
    NDCG accounts for the position of relevant items in the recommendation list,
    rewarding models that place relevant items at higher ranks.
    """
    dcg = 0.0
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)
            
    idcg = 0.0
    # Calculate Ideal DCG (all top positions filled by relevant items)
    for i in range(min(len(recommended_items), len(relevant_items))):
        idcg += 1.0 / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0


# ==============================================================================
# 2. RANKING EVALUATION PIPELINE
# ==============================================================================

def evaluate_ranking_models(model_predictions, k=10, relevance_threshold=4.0):
    """
    Iterates through all models in the predictions dictionary and calculates
    aggregated ranking metrics across all users.
    
    Args:
        model_predictions (dict): Dictionary where keys are model names and 
                                  values are DataFrames containing predicted vs actual ratings.
        k (int): The number of top recommendations to consider.
        relevance_threshold (float): Minimum actual rating for an item to be considered 'relevant'.
    """
    if not isinstance(model_predictions, dict):
        logger.error("Input 'model_predictions' is not a dictionary.")
        return None

    final_metrics = {}
    
    for model_name, predictions_df in model_predictions.items():
        if not isinstance(predictions_df, pd.DataFrame):
            logger.warning(f"Skipping '{model_name}': Result is not a DataFrame.")
            continue
            
        # Initialize per-user metric storage
        user_metrics = defaultdict(lambda: {'precision': 0.0, 'recall': 0.0, 'ndcg': 0.0, 'hit_rate': 0.0})
        
        # Identify ground-truth relevant items per user
        relevant_items_per_user = (
            predictions_df[predictions_df['actual_rating'] >= relevance_threshold]
            .groupby('user_id')['item_id'].apply(set).to_dict()
        )
        
        # Get top-K recommendations based on predicted scores
        top_k_predictions = (
            predictions_df.sort_values(by=['user_id', 'predicted_rating'], ascending=[True, False])
            .groupby('user_id')['item_id'].apply(lambda x: list(x[:k])).to_dict()
        )
        
        users_count = 0
        
        for user_id in relevant_items_per_user.keys():
            relevant = relevant_items_per_user.get(user_id, set())
            recommended = top_k_predictions.get(user_id, [])
            
            if relevant:
                prec, rec = get_precision_recall(recommended, relevant)
                ndcg = get_ndcg(recommended, relevant)
                hit_rate = get_hit_rate(recommended, relevant)
                
                user_metrics[user_id]['precision'] = prec
                user_metrics[user_id]['recall'] = rec
                user_metrics[user_id]['ndcg'] = ndcg
                user_metrics[user_id]['hit_rate'] = hit_rate
                users_count += 1

        # Calculate Mean Average Metrics
        if users_count > 0:
            avg_metrics = {
                f'Precision@{k}': sum(m['precision'] for m in user_metrics.values()) / users_count,
                f'Recall@{k}': sum(m['recall'] for m in user_metrics.values()) / users_count,
                f'NDCG@{k}': sum(m['ndcg'] for m in user_metrics.values()) / users_count,
                f'Hit Rate@{k}': sum(m['hit_rate'] for m in user_metrics.values()) / users_count,
            }
        else:
            avg_metrics = {f'Precision@{k}': 0, f'Recall@{k}': 0, f'NDCG@{k}': 0, f'Hit Rate@{k}': 0}
            
        final_metrics[model_name] = avg_metrics
        
    return pd.DataFrame(final_metrics).T.rename_axis('Model').reset_index()


# ==============================================================================
# 3. FINAL OUTPUT EXECUTION
# ==============================================================================

def run_evaluation_pipeline(model_results):
    """
    Orchestrates the evaluation process using settings from the configuration file.
    Logs the final results table for easy performance comparison.
    """
    # Retrieve parameters from Config
    k_val = config['evaluation']['k']
    threshold_val = config['evaluation']['relevance_threshold']
    
    if not model_results:
        logger.warning("No ranking data found to evaluate.")
        return None

    logger.info(f"Calculating Ranking Metrics (Top-{k_val}, Threshold >= {threshold_val})...")
    
    # Execute core evaluation logic
    ranking_results = evaluate_ranking_models(model_results, k=k_val, relevance_threshold=threshold_val)

    # Log results in a formatted table
    logger.info("Evaluation Results:")
    results_str = ranking_results.to_string(index=False, float_format="%.4f")
    logger.info("\n" + results_str)
    
    return ranking_results