"""
Gnosys Pipeline Orchestrator
----------------------------
This is the central execution script for the Gnosys Recommendation System. 
It coordinates the sequence of operations: initialization, data preparation, 
model training (Content-Based and Collaborative Filtering), and performance evaluation.
"""

import utils
import data_preparation 
import models 
import evaluation

def main():
    """
    Main entry point for the Gnosys Pipeline.
    Manages the flow of data through preparation, modeling, and evaluation stages.
    """
    
    # -----------------------------------------------------------------
    # 1. INITIALIZATION (Config & Logging)
    # -----------------------------------------------------------------
    # Load global settings from config.json and activate the logging system
    config = utils.load_config()
    logger = utils.setup_logging(config)
    logger.info("Starting Gnosys Execution Pipeline 📚")

    
    # -----------------------------------------------------------------
    # 2. DATA PREPARATION STAGE
    # -----------------------------------------------------------------
    try:
        # data_preparation handles path resolution and caching internally.
        # This stage generates the dataframes and sparse matrices required for modeling.
        books_df, ratings_df, ratings_matrix, user_to_row_id, book_to_col_id = data_preparation.run_data_preparation()
        logger.info("Data Preparation stage completed successfully.")
        
    except FileNotFoundError as e:
        logger.error(f"Required dataset file missing: {e}")
        return
    except Exception as e:
        # exc_info=True captures the full traceback in the log file for debugging
        logger.critical(f"Unexpected error during Data Preparation: {e}", exc_info=True)
        return
    
    # -----------------------------------------------------------------
    # 3. MODELING STAGE (Content-Based & Collaborative Filtering)
    # -----------------------------------------------------------------
    
    logger.info("🧠 Initializing Modeling Pipeline")
    
    try:
        # Execution of recommendation algorithms.
        # Active models and hyperparameters are defined in config.json.
        models_assets = models.run_models_pipeline(
            books_df, ratings_df, ratings_matrix, user_to_row_id, book_to_col_id
        )
    except Exception as e:
        logger.critical(f"Critical error during Modeling phase: {e}", exc_info=True)
        return
    
    # -----------------------------------------------------------------
    # 4. EVALUATION STAGE (Accuracy & Ranking Metrics)
    # -----------------------------------------------------------------
    
    # Check execution flags to determine if the evaluation suite should run
    run_eval = config.get("execution_flags", {}).get("run_evaluation", True)
    
    if run_eval:
        logger.info("📈 Initializing Evaluation Pipeline")
        try:
            # Assesses model performance using the predictions generated in the previous stage
            evaluation.run_evaluation_pipeline(models_assets['predictions'])
        except Exception as e:
            logger.error(f"Error encountered during Evaluation: {e}", exc_info=True)
            return
    else:
        logger.info("⏩ Skipping Evaluation Pipeline (flag disabled in config).")
        
    # -----------------------------------------------------------------
    # 5. FINALIZATION
    # -----------------------------------------------------------------
    
    logger.info("✅ All Pipeline Stages Completed Successfully!")
    
    
if __name__ == "__main__":
    main()