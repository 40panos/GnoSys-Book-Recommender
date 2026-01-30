"""
Gnosys Utility Module
---------------------
This module provides foundational helper functions for the Gnosys ecosystem, 
specifically managing configuration loading and centralized logging 
initialization for both console and file output.
"""

import logging
import os
import json
import sys

def load_config(config_path="config.json"):
    """
    Loads application settings from a JSON configuration file.
    
    If the file is missing or contains invalid JSON, the program logs a 
    critical error to the console and performs a clean exit.

    Args:
        config_path (str): The relative path to the configuration file.

    Returns:
        dict: The parsed configuration parameters.
    """
    if not os.path.exists(config_path):
        # Using print here as logging system is not yet initialized
        print(f"[CRITICAL] Configuration file '{config_path}' not found.")
        sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        print(f"[CRITICAL] Error parsing '{config_path}': {e}")
        sys.exit(1)


def setup_logging(config):
    """
    Configures the centralized logging system for the entire application.
    
    Sets up two separate handlers:
    1. A Console Handler for real-time monitoring.
    2. A File Handler for detailed debugging and audit trails.
    
    Includes timestamped formatting and automatic directory creation for log files.

    Args:
        config (dict): Configuration dictionary containing logging settings.

    Returns:
        logging.Logger: The configured root logger instance.
    """
    
    # 1. Retrieve logging parameters from configuration
    log_settings = config.get("logging", {})
    log_file_path = log_settings.get("log_file", "logs/app.log")
    
    # Retrieve and parse logging levels (e.g., INFO, DEBUG)
    console_level_str = log_settings.get("console_level", "INFO").upper()
    file_level_str = log_settings.get("file_level", "DEBUG").upper()
    
    console_level = getattr(logging, console_level_str, logging.INFO)
    file_level = getattr(logging, file_level_str, logging.DEBUG)

    # 2. Ensure the log directory exists
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 3. Initialize Root Logger
    logger = logging.getLogger()
    # Root level captures everything; specific handlers will filter based on their own levels
    logger.setLevel(logging.DEBUG) 
    
    # Clear existing handlers to prevent duplicate log entries during re-runs
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- A. CONSOLE HANDLER (Standard Output) ---
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(console_level)
    
    # Format: [YYYY-MM-DD HH:MM:SS] [LEVEL] Message
    c_format = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # --- B. FILE HANDLER (Persistent Log) ---
    # Set mode='w' to overwrite the log file on each fresh execution
    f_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    f_handler.setLevel(file_level)
    
    # Detailed Format: timestamp | MODULE | LEVEL | Message
    f_format = logging.Formatter('%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    return logger