"""
Logging utilities for holopy.
"""
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from ..config.constants import (
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_FILE_PATH,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    LOG_TO_CONSOLE,
    LOG_TO_FILE
)

def get_logger(
    name: str,
    level: Optional[int] = None,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Get a logger with specified configuration.
    
    Args:
        name: Logger name (usually __name__)
        level: Optional logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Only configure if no handlers exist
    if not logger.handlers:
        # Set level
        logger.setLevel(level or LOG_LEVEL)
        
        # Create formatter
        formatter = logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt=LOG_DATE_FORMAT
        )
        
        # Add console handler if enabled
        if LOG_TO_CONSOLE:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Add file handler if enabled
        if LOG_TO_FILE:
            try:
                # Create log directory if needed
                log_path = Path(log_file or LOG_FILE_PATH)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create rotating file handler
                file_handler = RotatingFileHandler(
                    filename=str(log_path),
                    maxBytes=LOG_MAX_BYTES,
                    backupCount=LOG_BACKUP_COUNT,
                    encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                # Fallback to console logging if file handling fails
                console_handler = logging.StreamHandler(sys.stderr)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
                logger.error(f"Failed to setup file logging: {str(e)}")
    
    return logger

class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter with context information."""
    
    def __init__(self, logger: logging.Logger, extra: dict = None):
        """Initialize adapter with extra context."""
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        """Process log message with context."""
        # Add timestamp
        timestamp = datetime.now().strftime(LOG_DATE_FORMAT)
        
        # Add context from extra dict
        context = [f"{k}={v}" for k, v in self.extra.items()]
        context_str = " ".join(context)
        
        # Format message
        if context_str:
            msg = f"[{timestamp}] {msg} ({context_str})"
        else:
            msg = f"[{timestamp}] {msg}"
            
        return msg, kwargs

def get_context_logger(
    name: str,
    context: dict = None
) -> LoggerAdapter:
    """
    Get a logger with context information.
    
    Args:
        name: Logger name
        context: Optional context dictionary
        
    Returns:
        Logger adapter with context
    """
    logger = get_logger(name)
    return LoggerAdapter(logger, context) 