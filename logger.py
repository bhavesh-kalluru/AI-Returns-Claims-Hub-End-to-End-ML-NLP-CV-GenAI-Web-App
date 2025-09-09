from loguru import logger
import sys

def get_logger():
    # Basic console logger; can expand later (rotate, serialize, etc.)
    logger.remove()
    logger.add(sys.stdout, level="INFO", enqueue=True, backtrace=False, diagnose=False)
    return logger
