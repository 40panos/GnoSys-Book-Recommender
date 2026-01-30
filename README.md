GnoSys Book Recommendation Ecosystem
GnoSys is a professional-grade book recommendation engine that blends traditional Machine Learning techniques with modern Generative AI. It features a sophisticated hybrid recommendation pipeline and an AI-powered Librarian that understands natural language queries.
Key Features
Hybrid Recommendation Engine: Combines Content-Based Filtering (TF-IDF on descriptions and tags) with Collaborative Filtering (User-Item interaction via Truncated SVD).

AI Librarian: An interactive chatbot powered by OpenAI's GPT-4o-mini that translates user moods and plot descriptions into database search terms.

Advanced Analytics: Evaluation suite featuring RMSE/MAE for prediction accuracy and NDCG, Precision@K, and Recall@K for ranking quality.

Interactive UI: A sleek, dark-themed dashboard built with Streamlit for browsing, searching, and chatting.

Optimized Performance: Integrated caching system (Pickle/Joblib) for pre-processed datasets and similarity matrices to ensure near-instant responses.

File,Description
main.py,The core Orchestrator that runs the entire pipeline from data to evaluation.
data_preparation.py,"Handles ETL processes, data cleaning, and sparse matrix generation."
models.py,"Contains the ML algorithms (SVD, TF-IDF, Demographic Filtering)."
ai_assistant.py,The LLM Backend logic for intent analysis and keyword extraction.
evaluation.py,Specialized module for Performance Metrics and model validation.
main_app.py,The primary Streamlit UI integrating the library and the AI Assistant.
utils.py,Shared utilities for JSON Configuration and centralized Logging.
config.json,Centralized control for all hyperparameters and execution flags.