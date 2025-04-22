# In src/utils/logger_config.py
import logging
import os

# Keep track of run_id across modules
_current_run_id = None


def set_run_id(run_id):
    global _current_run_id
    _current_run_id = run_id


def get_logger():
    # Create logs directory if it doesn't exist
    os.makedirs("../logs", exist_ok=True)

    # Use the current run_id
    global _current_run_id
    run_id = _current_run_id

    # Name the log file based on run_id or use default
    filename = f"../logs/{run_id}.log" if run_id else "../logger.log"

    # Get a logger with the run_id as name
    logger = logging.getLogger(str(run_id))

    # Configure the logger if it hasn't been configured yet
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(filename, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
