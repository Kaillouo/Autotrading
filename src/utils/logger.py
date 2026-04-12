import logging
import os

# Absolute path to logs/ — safe regardless of CWD (Task Scheduler, etc.)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOG_PATH = os.path.join(_PROJECT_ROOT, "logs", "trading.log")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    fh = logging.FileHandler(_LOG_PATH)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    return logger
