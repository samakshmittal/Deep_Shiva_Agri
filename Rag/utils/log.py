# logger_utils.py
import logging

def setup_logger(name="pipeline_logger", log_file="pipeline.log", level=logging.INFO):
    """Creates and returns a configured logger."""
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', "%H:%M:%S")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
