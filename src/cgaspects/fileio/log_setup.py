import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_log_file_path():
    """Get the path to the log file, creating directories if needed."""
    log_dir = Path.home() / ".crystalgrower" / "cgaspects"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "report.log"


def setup_logging(basic="DEBUG", console="INFO"):
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level=basic)
    # Setting matplotlib logger to warning
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a rotating file handler to write logs to a file
    # Max size: 10MB, keep 3 backup files
    log_file = get_log_file_path()
    file_handler = RotatingFileHandler(
        log_file,
        mode="a",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(basic)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s-%(name)s-%(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # Create a console handler to write logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console)
    console_formatter = logging.Formatter("%(name)s-%(levelname)s: %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)

    # Get the root logger and add both the file and console handlers to it
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
