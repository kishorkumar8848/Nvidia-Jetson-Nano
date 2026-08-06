import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from config.config import settings

# Prevent Unicode encoding crashes on Windows consoles by using backslashreplace
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(errors='backslashreplace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(errors='backslashreplace')
    except Exception:
        pass

# Ensure logs directory exists
LOGS_DIR = Path(settings.get_absolute_path("logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Path to log file
LOG_FILE_PATH = LOGS_DIR / "app.log"

def setup_logger(name: str = "HealthcareAI") -> logging.Logger:
    """Configures and returns a thread-safe rotating file and console logger."""
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    # Resolve log level from settings
    log_level_str = settings.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    # Standard formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating File handler (max 10MB per file, keeping up to 5 backups)
    try:
        file_handler = RotatingFileHandler(
            LOG_FILE_PATH,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to set up file logging: {e}", file=sys.stderr)

    return logger

# Export a default app logger
logger = setup_logger()
