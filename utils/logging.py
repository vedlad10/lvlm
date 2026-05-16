"""Logging utilities for LVLM."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import sys


# Global logger registry
_loggers = {}


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    format_str: Optional[str] = None
) -> logging.Logger:
    """Setup logger with console and optional file handler.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional filename for log file (default: {name}.log)
        log_dir: Directory to save log file (creates if not exists)
        format_str: Custom format string (default: standard format)
        
    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_str)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            log_file = f"{name}.log"
        
        log_path = log_dir / log_file
        
        # Rotating file handler to prevent huge log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10_000_000,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    _loggers[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger by name. If not found, returns root logger.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    
    return logger


class Logger:
    """Context manager for logging function execution time and calls."""
    
    def __init__(self, logger: logging.Logger, level: str = "INFO"):
        """Initialize logger context manager.
        
        Args:
            logger: Logger instance
            level: Logging level
        """
        self.logger = logger
        self.level = level
        self.start_time = None
    
    def __call__(self, func):
        """Decorator for logging function calls."""
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log_fn = getattr(self.logger, self.level.lower())
            log_fn(f"Starting: {func.__name__}")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                log_fn(f"Completed: {func.__name__} ({elapsed:.2f}s)")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(f"Failed: {func.__name__} ({elapsed:.2f}s) - {str(e)}")
                raise
        
        return wrapper


def log_multiple_lines(logger: logging.Logger, lines: list, level: str = "INFO") -> None:
    """Log multiple lines with consistent formatting.
    
    Args:
        logger: Logger instance
        lines: List of lines to log
        level: Logging level
    """
    log_fn = getattr(logger, level.lower())
    for line in lines:
        log_fn(line)


def create_summary_logger(logger: logging.Logger, title: str, data: dict) -> None:
    """Create formatted summary log.
    
    Args:
        logger: Logger instance
        title: Summary title
        data: Dictionary of key-value pairs to log
    """
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)
    
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            logger.info(f"{key}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    logger.info(f"  {k}: {v}")
            elif isinstance(value, list):
                for item in value:
                    logger.info(f"  - {item}")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info("=" * 60)
