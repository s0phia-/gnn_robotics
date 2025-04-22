import logging
import os


# Keep track of run_id across modules
_current_run_id = None


def set_run_id(run_id):
    global _current_run_id
    _current_run_id = run_id


def get_logger(run_id=None, run_dir=None):
    if run_dir and run_id:
        # Create logs directory within the run directory
        log_dir = f"{run_dir}/logs"
        os.makedirs(log_dir, exist_ok=True)

        # Name the log file based on run_id
        filename = f"{log_dir}/{run_id}.log"

        # Get a logger with the run_id as name
        logger = logging.getLogger(str(run_id))

        # Avoid adding handlers if this logger already has them
        if not logger.handlers:
            # Set level
            logger.setLevel(logging.DEBUG)

            # Create file handler
            file_handler = logging.FileHandler(filename, mode='w')

            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            # Add handler to logger
            logger.addHandler(file_handler)
    else:
        # Fallback to default logger
        logger = logging.getLogger("default")
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    return logger
