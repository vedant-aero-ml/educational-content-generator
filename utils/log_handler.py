import os
import logging
from contextlib import contextmanager

from config import LOG_DIR


@contextmanager
def request_logger(request_id: str):
    """Context manager that attaches a file handler to the root logger for the request duration.

    All log statements (from main.py, generator, evaluation, retrieval, etc.)
    are captured into logs/{request_id}.log while inside this context.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{request_id}.log")

    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        yield log_path
    finally:
        handler.flush()
        handler.close()
        root_logger.removeHandler(handler)
