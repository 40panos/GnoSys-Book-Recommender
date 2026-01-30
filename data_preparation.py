"""
Gnosys Data Preparation Module
------------------------------
This module handles data loading, cleaning, and preprocessing for books, 
tags, and user ratings. It includes a robust caching mechanism to 
optimize execution time during repetitive runs.
"""

import pandas as pd
from ast import literal_eval
import numpy as np
import time 
import os
import scipy.sparse as sp
import pickle
import logging
import utils

# --- SETUP LOGGER & CONFIG ---
logger = logging.getLogger(__name__)
config = utils.load_config()

# --- PATHS DEFINITION (FROM CONFIG) ---
BOOKS_PATH      = config['paths']['books']
BOOK_TAGS_PATH  = config['paths']['book_tags']
TAGS_PATH       = config['paths']['tags']
RATINGS_PATH    = config['paths']['ratings']
CACHE_PATH      = config['paths']['cache_prep']


# ==============================================================================
# 1. DATA LOADING FUNCTIONS
# ==============================================================================

def load_datasets():
    """
    Loads raw CSV datasets from the paths defined in the configuration file.
    
    Returns:
        tuple: DataFrames for books, book_tags, tags, and ratings.
    """
    logger.debug(f"Loading datasets from: {os.path.dirname(BOOKS_PATH)}")
    try:
        books      = pd.read_csv(BOOKS_PATH)
        book_tags  = pd.read_csv(BOOK_TAGS_PATH)
        tags       = pd.read_csv(TAGS_PATH)
        ratings    = pd.read_csv(RATINGS_PATH)
        return books, book_tags, tags, ratings
    except FileNotFoundError as e:
        logger.critical(f"Dataset file not found: {e}")
        raise


# ==============================================================================
# 2. FEATURE ENGINEERING & CLEANING
# ==============================================================================

def prepare_tags(book_tags, tags):
    """
    Merges tag names with book IDs and aggregates them into a single 
    space-separated string for each book to facilitate text-based search.
    """
    logger.debug("Processing tags...")
    merged = pd.merge(book_tags, tags, on="tag_id", how="left")
    
    grouped = (
        merged.groupby("goodreads_book_id")["tag_name"]
        .apply(lambda x: " ".join([str(tag) for tag in x if pd.notna(tag)]))
        .reset_index()
        .rename(columns={"tag_name": "all_tags"})
    )
    return grouped


def clean_list_column(x):
    """
    Helper function to sanitize columns containing lists (e.g., genres, authors).
    Converts items to lowercase and removes whitespaces for consistency.
    """
    if isinstance(x, str):
        try:
            lst = literal_eval(x)
            if isinstance(lst, list):
                return [str(i).lower().replace(" ", "") for i in lst]
        except (ValueError, SyntaxError):
            return []
    elif isinstance(x, list):
        return [str(i).lower().replace(" ", "") for i in x]
    return []


def prepare_books(books, all_tags_df):
    """
    Primary book dataset cleaning function. Handles merging with tags, 
    dropping irrelevant columns, managing null values, and creating 
    composite 'meta_features' for metadata-based filtering.
    """
    logger.debug("Processing books features (cleaning & merging)...")
    
    books_merged = pd.merge(
        books,
        all_tags_df,
        on="goodreads_book_id",
        how="left"
    )

    # Remove columns not required for modeling or UI display
    cols_to_drop = [
        "Unnamed: 0", "index", "best_book_id", "books_count",
        "isbn", "isbn13", "language_code", "original_publication_year",
        "work_id", "work_ratings_count", "work_text_reviews_count",
        "authors_2", "original_title", "publishDate",
        "ratings_1", "ratings_2", "ratings_3", "ratings_4", "ratings_5",
    ]
    books_merged = books_merged.drop(columns=cols_to_drop, errors="ignore")

    books_merged["description"] = books_merged.get("description", pd.Series()).fillna("").astype(str)
    books_merged["all_tags"] = books_merged.get("all_tags", pd.Series()).fillna("").astype(str)

    if "title" in books_merged.columns:
        books_merged["title_clean"] = books_merged["title"].astype(str).str.strip().str.lower()

    # Apply list cleaning to categorical features
    for col in ["authors", "genres"]:
        if col in books_merged.columns:
            books_merged[col] = books_merged[col].apply(clean_list_column)
        else:
            books_merged[col] = [[]]

    # Concatenate features to create a unified metadata string
    books_merged["meta_features"] = books_merged.apply(
        lambda x: ",".join(x["genres"]) + "," + ",".join(x["authors"]),
        axis=1
    )

    return books_merged


# ==============================================================================
# 3. RATINGS PREPARATION & MATRIX GENERATION
# ==============================================================================

def prepare_ratings(ratings, min_ratings_per_user=5, min_ratings_per_book=5):
    """
    Filters user ratings based on activity thresholds and generates a 
    CSR Sparse Matrix for Collaborative Filtering.
    """
    logger.debug(f"Filtering ratings (Min User: {min_ratings_per_user}, Min Book: {min_ratings_per_book})...")
    
    ratings = ratings.drop_duplicates(subset=["user_id", "book_id"])

    # Filter out inactive users to reduce noise
    user_counts = ratings["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_ratings_per_user].index
    ratings = ratings[ratings["user_id"].isin(valid_users)]

    # Filter out niche books with insufficient data
    book_counts = ratings["book_id"].value_counts()
    valid_books = book_counts[book_counts >= min_ratings_per_book].index
    ratings = ratings[ratings["book_id"].isin(valid_books)]

    # Pivot the data into a User-Item interaction matrix
    logger.debug("Creating Pivot Table (Dense)...")
    ratings_matrix_dense = ratings.pivot_table(
        index="user_id",
        columns="book_id",
        values="rating"
    )

    # Generate index mappings for internal matrix operations
    user_to_row_id = {id: i for i, id in enumerate(ratings_matrix_dense.index)}
    book_to_col_id = {id: i for i, id in enumerate(ratings_matrix_dense.columns)}
    
    # Convert to Compressed Sparse Row (CSR) format for memory efficiency
    logger.debug("Converting to CSR Sparse Matrix...")
    ratings_matrix_sparse = sp.csr_matrix(ratings_matrix_dense.fillna(0).values)
    
    return ratings, ratings_matrix_sparse, user_to_row_id, book_to_col_id


# ==============================================================================
# 4. DIAGNOSTICS & CACHING PIPELINE
# ==============================================================================

def log_data_diagnostics(ratings_df, total_books_count=0):
    """
    Logs statistical insights regarding the dataset size and matrix sparsity.
    """
    n_users = ratings_df['user_id'].nunique()
    n_books = ratings_df['book_id'].nunique()
    n_ratings = len(ratings_df)
    
    matrix_size = n_users * n_books
    sparsity = (1 - (n_ratings / matrix_size)) * 100 if matrix_size > 0 else 0
    
    logger.info("--- Data Diagnostics ---")
    logger.info(f"Filtered Ratings: {n_ratings}")
    logger.info(f"Filtered Users:   {n_users}")
    logger.info(f"Filtered Books:   {n_books}")
    logger.info(f"Sparsity:         {sparsity:.4f}%")
    
    # Analyze rating distribution
    dist = ratings_df['rating'].value_counts(normalize=True).sort_index()
    dist_str = ", ".join([f"{r}: {freq*100:.1f}%" for r, freq in dist.items()])
    logger.debug(f"Rating Distribution: {dist_str}")


def run_data_preparation():
    """
    Orchestrates the entire data preparation pipeline. Checks for existing 
    cache files to skip processing unless a force reload is requested.
    """
    force_reload = config.get('execution_flags', {}).get('force_data_reload', False)
    
    # --- ATTEMPT CACHE RETRIEVAL ---
    if os.path.exists(CACHE_PATH) and not force_reload:
        logger.info("Cache found. Loading pre-processed data...")
        start_time = time.time()
        try:
            with open(CACHE_PATH, 'rb') as f:
                data_bundle = pickle.load(f)
            
            books_prepared = data_bundle['books_prepared']
            ratings_filtered = data_bundle['ratings_filtered']
            ratings_matrix = data_bundle['ratings_matrix']
            user_to_row_id = data_bundle['user_to_row_id']
            book_to_col_id = data_bundle['book_to_col_id']
            
            logger.info(f"Data loaded from cache in {time.time() - start_time:.2f}s")
            log_data_diagnostics(ratings_filtered)
            
            return books_prepared, ratings_filtered, ratings_matrix, user_to_row_id, book_to_col_id
            
        except Exception as e:
            logger.warning(f"Cache corrupted ({e}). Proceeding to re-build data.")
    
    # --- START FULL PIPELINE ---
    if force_reload:
        logger.info("Force Reload is enabled. Re-building data...")
    else:
        logger.info("Cache not found or corrupted. Building data...")
        
    start_time_total = time.time()
    
    # 1. Load raw data
    books, book_tags, tags, ratings = load_datasets() 
    
    # 2. Prepare feature-rich tags
    all_tags_df = prepare_tags(book_tags, tags)
    
    # 3. Clean and merge book attributes
    books_prepared = prepare_books(books, all_tags_df)
    
    # 4. Filter ratings and generate interaction matrix
    min_u = config['preprocessing']['min_ratings_per_user']
    min_b = config['preprocessing']['min_ratings_per_book']
    
    ratings_filtered, ratings_matrix, user_to_row_id, book_to_col_id = prepare_ratings( 
        ratings, 
        min_ratings_per_user=min_u, 
        min_ratings_per_book=min_b
    )
    
    elapsed_time = time.time() - start_time_total
    logger.info(f"Data Preparation completed in {elapsed_time:.2f}s")
    
    log_data_diagnostics(ratings_filtered)

    # 5. Serialize processed data to cache
    logger.info("Saving data to cache...")
    try:
        data_to_cache = {
            'books_prepared': books_prepared,
            'ratings_filtered': ratings_filtered,
            'ratings_matrix': ratings_matrix,
            'user_to_row_id': user_to_row_id,
            'book_to_col_id': book_to_col_id,
            'book_tags': book_tags,
            'tags': tags,
            'total_books_count': books['goodreads_book_id'].nunique()
        }
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(data_to_cache, f)
        logger.info("Cache saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

    return books_prepared, ratings_filtered, ratings_matrix, user_to_row_id, book_to_col_id