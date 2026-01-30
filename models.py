"""
Gnosys Recommender Engine - Models Module
-----------------------------------------
This module implements the core recommendation algorithms, including 
Demographic Filtering, Content-Based Filtering (TF-IDF), 
Collaborative Filtering (User/Item Memory-Based), and Matrix Factorization (SVD).
"""

import numpy as np
import time
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel 
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import joblib 
import os
import logging
import utils

# ==============================================================================
# INITIALIZATION & LOGGING SETUP
# ==============================================================================

logger = logging.getLogger(__name__)
config = utils.load_config()

# Define directory for serializing trained models and similarity matrices
MODELS_CACHE_DIR = config['paths']['cache_models_dir']

if not os.path.exists(MODELS_CACHE_DIR):
    os.makedirs(MODELS_CACHE_DIR)


def load_or_build_similarity(name, build_func, *args, **kwargs):
    """
    Caching wrapper for similarity matrices. 
    Retrieves serialized matrices from disk or computes them if they don't exist.
    """
    file_path = os.path.join(MODELS_CACHE_DIR, f"sim_matrix_{name}.pkl")
    
    if os.path.exists(file_path):
        logger.debug(f"Loading cached similarity matrix for: {name}")
        return joblib.load(file_path)
    else:
        logger.info(f"Building similarity matrix for: {name} (First run)...")
        start_time = time.time()
        matrix = build_func(*args, **kwargs)
        logger.debug(f"Matrix built in {time.time() - start_time:.2f}s")
        
        logger.debug(f"Saving matrix for {name}...")
        try:
            joblib.dump(matrix, file_path)
        except Exception as e:
            logger.warning(f"Could not save matrix {name}: {e}")
            
        return matrix


# ==============================================================================
# SECTION 1: DEMOGRAPHIC FILTERING
# ==============================================================================

def demographic_filtering(books_df, quantile=0.70):
    """
    Implements Weighted Rating (IMDB Formula). 
    Calculates a score based on popularity and average rating to identify 
    globally 'Trending' books.
    """
    C = books_df["average_rating"].mean()
    m = books_df["ratings_count"].quantile(quantile)

    qualified = books_df[books_df["ratings_count"] >= m].copy()

    def weighted_rating(x):
        v = x["ratings_count"]
        R = x["average_rating"]
        return (v/(v+m) * R) + (m/(m+v) * C)

    qualified["score"] = qualified.apply(weighted_rating, axis=1)
    
    # Sort and return only the indices of the top-qualified books
    top_indices = qualified.sort_values("score", ascending=False).index.tolist()
    
    return top_indices


# ==============================================================================
# SECTION 2: SIMILARITY & SPARSE MATRIX HELPERS
# ==============================================================================

def build_similarity_matrix(series, method='tfidf'):
    """
    Generates a Cosine Similarity matrix from a text series.
    """
    if method == 'tfidf':
        vec = TfidfVectorizer(stop_words='english')
    else:
        vec = CountVectorizer()

    matrix = vec.fit_transform(series.fillna(''))
    sim = cosine_similarity(matrix)
    return sim


def get_top_similar(idx, sim_matrix, n=10):
    """
    Retrieves the 'n' most similar item indices for a given index from a similarity matrix.
    """
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:n+1]]
    return top_indices


def center_sparse_rows(matrix):
    """
    Performs Mean Centering on a Sparse Matrix. 
    Subtracts the row mean from non-zero elements (useful for Pearson Correlation).
    """
    mat = matrix.copy().astype(float)
    counts = mat.getnnz(axis=1)
    sums = mat.sum(axis=1).A1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        means = np.divide(sums, counts)
        means[np.isnan(means)] = 0.0 
    
    coo = mat.tocoo()
    mat.data -= means[coo.row]
    
    return mat.tocsr()


# ==============================================================================
# SECTION 3: CONTENT-BASED FILTERING
# ==============================================================================

def build_tfidf_similarity(series, min_df=3, max_df=1.0, stop_words='english'):
    """
    Computes Cosine Similarity using TF-IDF Vectorization for text-based features.
    """
    tfidf_vec = TfidfVectorizer(
        stop_words=stop_words,
        min_df=min_df,       
        max_df=max_df       
    )
    
    sparse_matrix = tfidf_vec.fit_transform(series.fillna(''))
    similarity_matrix = linear_kernel(sparse_matrix, sparse_matrix)
    return similarity_matrix


def build_count_similarity(series, stop_words='english', token_pattern=r'\b\w{2,}\b'):
    """
    Computes Cosine Similarity using Count Vectorization for metadata/categorical features.
    """
    count_vec = CountVectorizer(
        stop_words=stop_words,
        token_pattern=token_pattern
    )
    
    sparse_matrix = count_vec.fit_transform(series.fillna(''))
    similarity_matrix = linear_kernel(sparse_matrix, sparse_matrix)
    return similarity_matrix


def build_mixed_similarity(sim_matrices, weights):
    """
    Combines multiple similarity matrices (e.g., Description + Tags) 
    using a Weighted Sum approach.
    """
    if len(sim_matrices) != len(weights):
        raise ValueError("Number of matrices must match number of weights.")
    
    logger.debug(f"Mixing {len(sim_matrices)} matrices with weights: {weights}")
    
    mixed_sim = sim_matrices[0] * weights[0]
    for i in range(1, len(sim_matrices)):
        mixed_sim += sim_matrices[i] * weights[i]
        
    return mixed_sim


# ==============================================================================
# SECTION 4: COLLABORATIVE FILTERING (MEMORY-BASED)
# ==============================================================================

def generate_memory_cf_predictions(mode, train_matrix, test_df, user_map, item_map, top_k=20, user_limit=10000, metric='cosine'):
    """
    Generates User-User or Item-Item Memory-Based Collaborative Filtering predictions.
    """
    logger.info(f"Generating {mode.upper()}-Based ({metric}) predictions...")
    
    # 1. Setup matrices based on mode (User vs Item)
    if mode == 'item':
        matrix_to_sim = train_matrix.T
        active_matrix = train_matrix 
        target_indices = np.arange(train_matrix.shape[0])
        
    elif mode == 'user':
        n_users = train_matrix.shape[0]
        if n_users > user_limit:
            logger.warning(f"User limit reached. Sampling first {user_limit} users.")
            active_matrix = train_matrix[:user_limit, :]
            target_indices = np.arange(user_limit)
        else:
            active_matrix = train_matrix
            target_indices = np.arange(n_users)
            
        matrix_to_sim = active_matrix
    else:
        raise ValueError("Mode must be 'user' or 'item'")

    # 2. Apply Row Centering for Pearson metric
    if metric == 'pearson':
        matrix_to_sim = center_sparse_rows(matrix_to_sim)
    
    # 3. Compute Similarity Matrix
    sim_matrix = cosine_similarity(matrix_to_sim, dense_output=False)
    
    # 4. Generate Prediction Scores
    if mode == 'item':
        pred_scores_matrix = active_matrix.dot(sim_matrix)
    else: # user
        pred_scores_matrix = sim_matrix.dot(active_matrix)

    # 5. Extract Top-K recommendations
    users_list = []
    items_list = []
    scores_list = []
    
    inv_item_map = {v: k for k, v in item_map.items()}
    inv_user_map = {v: k for k, v in user_map.items()}
    test_users_ids = set(test_df['user_id'].unique())
    
    for idx, u_idx in enumerate(target_indices):
        if u_idx not in inv_user_map: continue
        user_id = inv_user_map[u_idx]
        if user_id not in test_users_ids: continue
        
        row_vec = pred_scores_matrix[idx, :] 
        
        if hasattr(row_vec, "toarray"):
            user_scores = row_vec.toarray().flatten()
        else:
            user_scores = np.asarray(row_vec).flatten()
        
        # Mask already known interactions
        known_indices = train_matrix[u_idx, :].nonzero()[1]
        user_scores[known_indices] = 0
        
        top_indices = user_scores.argsort()[-top_k:][::-1]
        
        for item_idx in top_indices:
            score = user_scores[item_idx]
            if score > 0: 
                users_list.append(user_id)
                items_list.append(inv_item_map[item_idx])
                scores_list.append(score)
                
    pred_df = pd.DataFrame({
        'user_id': users_list,
        'item_id': items_list,
        'predicted_rating': scores_list
    })
    
    final_pred_df = pd.merge(
        pred_df, test_df,
        left_on=['user_id', 'item_id'], right_on=['user_id', 'book_id'],
        how='left'
    )
    final_pred_df.rename(columns={'rating': 'actual_rating'}, inplace=True)
    final_pred_df['actual_rating'] = final_pred_df['actual_rating'].fillna(0)
    
    return final_pred_df


# ==============================================================================
# SECTION 5: MATRIX FACTORIZATION (SVD)
# ==============================================================================

def generate_svd_predictions(train_matrix, test_df, user_map, item_map, n_components=50, top_k=20):
    """
    Generates ranking-based recommendations using Truncated SVD.
    """
    logger.info(f"Generating SVD Predictions (k={n_components})...")
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(train_matrix) 
    item_factors = svd.components_                 
    
    users_list = []
    items_list = []
    scores_list = []
    
    inv_item_map = {v: k for k, v in item_map.items()}
    inv_user_map = {v: k for k, v in user_map.items()}
    test_users_ids = set(test_df['user_id'].unique())
    
    for u_idx, user_vec in enumerate(user_factors):
        if u_idx not in inv_user_map: continue
        user_id = inv_user_map[u_idx]
        if user_id not in test_users_ids: continue
        
        predicted_ratings = user_vec.dot(item_factors)
        
        # Mask already known interactions
        known_indices = train_matrix[u_idx, :].nonzero()[1]
        predicted_ratings[known_indices] = 0
        
        top_indices = predicted_ratings.argsort()[-top_k:][::-1]
        
        for item_idx in top_indices:
            score = predicted_ratings[item_idx]
            users_list.append(user_id)
            items_list.append(inv_item_map[item_idx])
            scores_list.append(score)
            
    pred_df = pd.DataFrame({
        'user_id': users_list,
        'item_id': items_list,
        'predicted_rating': scores_list
    })
    
    final_pred_df = pd.merge(
        pred_df, test_df,
        left_on=['user_id', 'item_id'], right_on=['user_id', 'book_id'],
        how='left'
    )
    final_pred_df.rename(columns={'rating': 'actual_rating'}, inplace=True)
    final_pred_df['actual_rating'] = final_pred_df['actual_rating'].fillna(0)
    
    return final_pred_df


# ==============================================================================
# SECTION 6: PERFORMANCE METRICS (RMSE / MAE)
# ==============================================================================

def calculate_svd_metrics(train_matrix, test_df, user_map, item_map, n_components=50, metrics_config=None):
    """
    Calculates RMSE and MAE for SVD-based predictions using Mean Centering.
    """
    if metrics_config is None:
        metrics_config = {'rmse': True, 'mae': True}

    logger.info(f"Calculating SVD Metrics (k={n_components})...")
    
    # 1. Compute Global User Means
    mat = train_matrix.copy().astype(float)
    counts = mat.getnnz(axis=1)
    sums = mat.sum(axis=1).A1
    with np.errstate(divide='ignore', invalid='ignore'):
        user_means = np.divide(sums, counts)
        user_means[np.isnan(user_means)] = 0.0
    
    # 2. Fit Centered SVD
    matrix_centered = center_sparse_rows(train_matrix)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(matrix_centered) 
    item_factors = svd.components_.T                  
    
    # 3. Predict Test Ratings
    u_indices = []
    i_indices = []
    actual_ratings = []
    u_means_vec = []
    
    for row in test_df.itertuples():
        if row.user_id in user_map and row.book_id in item_map:
            u_idx = user_map[row.user_id]
            u_indices.append(u_idx)
            i_indices.append(item_map[row.book_id])
            actual_ratings.append(row.rating)
            u_means_vec.append(user_means[u_idx])
            
    if not actual_ratings: return {}
    
    u_indices = np.array(u_indices)
    i_indices = np.array(i_indices)
    y_true = np.array(actual_ratings)
    u_means_vec = np.array(u_means_vec)
    
    preds_dot = np.sum(user_factors[u_indices] * item_factors[i_indices], axis=1)
    preds = preds_dot + u_means_vec
    preds = np.clip(preds, 1.0, 5.0)
    
    results = {}
    if metrics_config.get('rmse', True):
        results['RMSE'] = sqrt(mean_squared_error(y_true, preds))
    if metrics_config.get('mae', True):
        results['MAE'] = mean_absolute_error(y_true, preds)
        
    return results


def calculate_memory_cf_metrics(mode, train_matrix, test_df, user_map, item_map, user_sample_size=5000, test_eval_limit=5000, metrics_config=None):
    """
    Computes RMSE and MAE for Memory-Based CF using efficient sampling.
    """
    if metrics_config is None:
        metrics_config = {'rmse': True, 'mae': True}

    logger.info(f"Calculating {mode.upper()}-Based Metrics...")
    
    # 1. Compute similarity and prediction components
    if mode == 'item':
        sim_matrix = cosine_similarity(train_matrix.T, dense_output=False)
        active_matrix = train_matrix
        numerator = active_matrix.dot(sim_matrix)
        binary_matrix = active_matrix.copy()
        binary_matrix.data = np.ones_like(binary_matrix.data)
        denominator = binary_matrix.dot(sim_matrix)
        limit = train_matrix.shape[0] 
        
    else: # User-Based
        n_users = train_matrix.shape[0]
        limit = min(n_users, user_sample_size)
        logger.debug(f"Using {limit} users for similarity calculation.")
        
        active_matrix = train_matrix[:limit, :]
        sim_matrix = cosine_similarity(active_matrix, dense_output=False)
        numerator = sim_matrix.dot(active_matrix)
        binary_matrix = active_matrix.copy()
        binary_matrix.data = np.ones_like(binary_matrix.data)
        denominator = sim_matrix.dot(binary_matrix)

    # 2. Evaluation Sampling
    if len(test_df) > test_eval_limit:
        logger.debug(f"Sampling {test_eval_limit} test records for evaluation...")
        eval_df = test_df.sample(n=test_eval_limit, random_state=42)
    else:
        eval_df = test_df

    y_true = []
    y_pred = []
    
    for row in eval_df.itertuples():
        u_id = row.user_id
        b_id = row.book_id
        rating = row.rating
        
        if u_id not in user_map or b_id not in item_map: continue
        u_idx = user_map[u_id]
        i_idx = item_map[b_id]
        
        if mode == 'user' and u_idx >= limit: continue
        
        try:
            num = numerator[u_idx, i_idx]
            den = denominator[u_idx, i_idx]
            
            if den > 0:
                pred_val = num / den
                pred_val = max(1.0, min(5.0, pred_val))
                y_true.append(rating)
                y_pred.append(pred_val)
                
        except IndexError:
            continue

    if len(y_true) == 0:
        return {}
        
    results = {}
    if metrics_config.get('rmse', True):
        results['RMSE'] = sqrt(mean_squared_error(y_true, y_pred))
    if metrics_config.get('mae', True):
        results['MAE'] = mean_absolute_error(y_true, y_pred)
    
    return results


# ==============================================================================
# SECTION 7: HYBRID ENSEMBLE & ORCHESTRATION
# ==============================================================================

def generate_predictions(model_name, similarity_matrix, train_matrix, test_df, user_map, item_map, top_k=20):
    """
    Generic predictor that converts any Item-Item similarity matrix into user recommendations.
    """
    logger.info(f"Generating predictions for {model_name}...")
    
    pred_scores_matrix = train_matrix.dot(similarity_matrix)
    
    users_list = []
    items_list = []
    scores_list = []
    
    inv_item_map = {v: k for k, v in item_map.items()}
    test_users_ids = test_df['user_id'].unique()
    
    for user_id in test_users_ids:
        if user_id not in user_map:
            continue
            
        u_idx = user_map[user_id]
        user_vector = pred_scores_matrix[u_idx, :]
        
        if hasattr(user_vector, "toarray"):
            user_scores = user_vector.toarray().flatten()
        else:
            user_scores = np.asarray(user_vector).flatten()
        
        known_indices = train_matrix[u_idx, :].nonzero()[1]
        user_scores[known_indices] = 0
        
        top_indices = user_scores.argsort()[-top_k:][::-1]
        
        for item_idx in top_indices:
            score = user_scores[item_idx]
            if score > 0:
                users_list.append(user_id)
                items_list.append(inv_item_map[item_idx])
                scores_list.append(score)
                
    pred_df = pd.DataFrame({
        'user_id': users_list,
        'item_id': items_list,
        'predicted_rating': scores_list
    })
    
    final_pred_df = pd.merge(
        pred_df, test_df,
        left_on=['user_id', 'item_id'], right_on=['user_id', 'book_id'],
        how='left'
    )
    final_pred_df.rename(columns={'rating': 'actual_rating'}, inplace=True)
    final_pred_df['actual_rating'] = final_pred_df['actual_rating'].fillna(0)
    
    return final_pred_df


def generate_ensemble_predictions(content_df, cf_df, weights, test_df, top_k=20):
    """
    Combines results from Content-Based and Collaborative Filtering using weighted averaging.
    """
    logger.info("Generating Ensemble Predictions...")
    
    # 1. Prepare sub-model DataFrames
    cont_sub = content_df[['user_id', 'item_id', 'predicted_rating']].rename(columns={'predicted_rating': 'score_content'})
    cf_sub = cf_df[['user_id', 'item_id', 'predicted_rating']].rename(columns={'predicted_rating': 'score_cf'})
    
    # 2. Join candidate sets
    merged = pd.merge(cont_sub, cf_sub, on=['user_id', 'item_id'], how='outer').fillna(0)
    
    # 3. Weighted Blending (Normalization for scale differences)
    w_c = weights.get('content', 0.5)
    w_cf = weights.get('cf', 0.5)
    
    merged['final_score'] = (merged['score_content'] * w_c) + ((merged['score_cf'] / 5.0) * w_cf)
    
    # 4. Filter Top-K per user
    final_recs = (
        merged.sort_values(['user_id', 'final_score'], ascending=[True, False])
        .groupby('user_id')
        .head(top_k)
    )
    
    final_pred_df = final_recs[['user_id', 'item_id', 'final_score']].rename(columns={'final_score': 'predicted_rating'})
    
    final_output = pd.merge(
        final_pred_df, test_df,
        left_on=['user_id', 'item_id'], right_on=['user_id', 'book_id'],
        how='left'
    )
    final_output.rename(columns={'rating': 'actual_rating'}, inplace=True)
    final_output['actual_rating'] = final_output['actual_rating'].fillna(0)
    
    return final_output


def run_models_pipeline(books_df, ratings_df, ratings_matrix, user_to_row_id, book_to_col_id):
    """
    Main orchestrator for the modeling pipeline. 
    Executes models based on active flags in the config.
    """
    # Parameters retrieval
    test_size_val = config['preprocessing']['test_size']
    rand_state_val = config['preprocessing']['random_state']
    
    cb_min_df = config['models']['content_based']['min_df']
    cb_max_df = config['models']['content_based']['max_df']
    mixed_w = config['models']['content_based']['mixed_weights']
    
    top_k_val = config['models']['collaborative']['top_k_recommendations']
    user_limit_val = config['models']['collaborative']['user_based_limit']
    svd_k_val = config['models']['collaborative']['svd_n_components']
    
    ensemble_cfg = config['models'].get('ensemble', {})
    metrics_toggle = config['evaluation'].get('metrics', {'rmse': True, 'mae': True})

    flags = config.get('execution_flags', {})
    eval_mode = flags.get('evaluation_mode', 'ranking') 
    do_content = flags.get('run_content_based', False)
    do_cf_simple = flags.get('run_collaborative_simple', False)
    do_svd = flags.get('run_collaborative_svd', False)
    
    active = config.get('active_models', {})

    logger.info(f"Evaluation Mode: {eval_mode.upper()}")
    
    start_time_total = time.time()
    predictions_dict = {} 
    
    # --- Split Train/Test Sets ---
    logger.info(f"Splitting Train/Test (Size: {test_size_val})...")
    train_df, test_df = train_test_split(
        ratings_df, 
        test_size=test_size_val, 
        random_state=rand_state_val, 
        stratify=ratings_df['user_id']
    )
    
    rows = [user_to_row_id[uid] for uid in train_df['user_id']]
    cols = [book_to_col_id[bid] for bid in train_df['book_id']]
    data = train_df['rating'].values
    shape = (len(user_to_row_id), len(book_to_col_id))
    train_matrix = csr_matrix((data, (rows, cols)), shape=shape)


    # --- Content-Based Block ---
    if do_content:
        logger.info("Building Content-Based Models...")
        
        sim_tags = load_or_build_similarity(
            "tags", 
            build_tfidf_similarity, 
            books_df['all_tags'], 
            min_df=cb_min_df, 
            max_df=cb_max_df
        )
        
        sim_desc = load_or_build_similarity(
            "desc", 
            build_tfidf_similarity, 
            books_df['description']
        )
        
        sim_meta = load_or_build_similarity(
            "meta", 
            build_count_similarity, 
            books_df['meta_features']
        )
        
        sim_mixed = build_mixed_similarity(
            [sim_desc, sim_tags], 
            mixed_w
        )
        
        if active.get('content_tags', True):
            predictions_dict['Tags'] = generate_predictions("Tags Only", sim_tags, train_matrix, test_df, user_to_row_id, book_to_col_id, top_k=top_k_val)
            
        if active.get('content_description', True):
            predictions_dict['Description'] = generate_predictions("Desc Only", sim_desc, train_matrix, test_df, user_to_row_id, book_to_col_id, top_k=top_k_val)
            
        if active.get('content_metadata', True):
            predictions_dict['Metadata'] = generate_predictions("Meta Only", sim_meta, train_matrix, test_df, user_to_row_id, book_to_col_id, top_k=top_k_val)
            
        if active.get('content_mixed', True):
            predictions_dict['ContentMixed'] = generate_predictions("ContentMixed", sim_mixed, train_matrix, test_df, user_to_row_id, book_to_col_id, top_k=top_k_val)


    # --- Collaborative Filtering Block ---
    if do_cf_simple:
        logger.info("Building Memory-Based CF Models...")
        
        if eval_mode == 'ranking':
            if active.get('cf_item_cosine', True):
                predictions_dict['CF_Item_Cosine'] = generate_memory_cf_predictions('item', train_matrix, test_df, user_to_row_id, book_to_col_id, top_k=top_k_val, metric='cosine')
            
            if active.get('cf_item_pearson', True):
                predictions_dict['CF_Item_Pearson'] = generate_memory_cf_predictions('item', train_matrix, test_df, user_to_row_id, book_to_col_id, top_k=top_k_val, metric='pearson')
            
            if active.get('cf_user_cosine', True):
                predictions_dict['CF_User_Cosine'] = generate_memory_cf_predictions('user', train_matrix, test_df, user_to_row_id, book_to_col_id, user_limit=user_limit_val, top_k=top_k_val, metric='cosine')
            
            if active.get('cf_user_pearson', True):
                predictions_dict['CF_User_Pearson'] = generate_memory_cf_predictions('user', train_matrix, test_df, user_to_row_id, book_to_col_id, user_limit=user_limit_val, top_k=top_k_val, metric='pearson')

        elif eval_mode == 'rmse':
            res_item = calculate_memory_cf_metrics('item', train_matrix, test_df, user_to_row_id, book_to_col_id, metrics_config=metrics_toggle)
            logger.info(f"[RESULT] Item-Based Metrics: {res_item}")
            
            res_user = calculate_memory_cf_metrics('user', train_matrix, test_df, user_to_row_id, book_to_col_id, user_sample_size=user_limit_val, metrics_config=metrics_toggle)
            logger.info(f"[RESULT] User-Based Metrics: {res_user}")


    # --- Matrix Factorization (SVD) Block ---
    if do_svd:
        if active.get('svd', True):
            logger.info("Building SVD Model...")
            
            if eval_mode == 'ranking':
                predictions_dict['SVD_MatrixFact'] = generate_svd_predictions(
                    train_matrix=train_matrix, test_df=test_df, 
                    user_map=user_to_row_id, item_map=book_to_col_id,
                    n_components=svd_k_val, 
                    top_k=top_k_val
                )
                
            elif eval_mode == 'rmse':
                res_svd = calculate_svd_metrics(train_matrix, test_df, user_to_row_id, book_to_col_id, n_components=svd_k_val, metrics_config=metrics_toggle)
                logger.info(f"[RESULT] SVD (k={svd_k_val}) Metrics: {res_svd}")
        else:
            logger.info("SVD Skipped (Active Flag is False)")

    # --- Ensemble Block ---
    if active.get('ensemble', False) and eval_mode == 'ranking':
        content_key = ensemble_cfg.get('content_key', 'ContentMixed')
        cf_key = ensemble_cfg.get('cf_key', 'SVD_MatrixFact')
        
        if content_key in predictions_dict and cf_key in predictions_dict:
            logger.info(f"Building Ensemble Model ({content_key} + {cf_key})...")
            
            predictions_dict['Ensemble'] = generate_ensemble_predictions(
                content_df=predictions_dict[content_key],
                cf_df=predictions_dict[cf_key],
                weights=ensemble_cfg.get('weights', {'content': 0.5, 'cf': 0.5}),
                test_df=test_df,
                top_k=top_k_val
            )
        else:
            logger.warning(f"Ensemble skipped: Missing {content_key} or {cf_key} in predictions.")


    elapsed_time = time.time() - start_time_total
    logger.info(f"Models Pipeline completed in {elapsed_time:.2f}s")
    
    return {'predictions': predictions_dict}