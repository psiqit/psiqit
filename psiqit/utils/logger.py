# psiqit/utils/logger.py

"""
Logger Module
Logging utilities for PSIQIT
"""

import logging
import sys
from typing import Optional, Union
from datetime import datetime


# ============================================================================
# LOGGER CONFIGURATION
# ============================================================================

# Color codes for terminal output
COLORS = {
    'DEBUG': '\033[36m',     # Cyan
    'INFO': '\033[32m',      # Green
    'WARNING': '\033[33m',   # Yellow
    'ERROR': '\033[31m',     # Red
    'CRITICAL': '\033[35m',  # Magenta
    'RESET': '\033[0m',      # Reset
}

# Log level names
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}


# ============================================================================
# COLORED FORMATTER
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter with colors for terminal output
    """
    
    def __init__(self, fmt: Optional[str] = None, use_colors: bool = True):
        """
        Initialize colored formatter
        
        Args:
            fmt: Format string
            use_colors: Enable colors in output
        """
        if fmt is None:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        super().__init__(fmt)
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors
        
        Args:
            record: Log record
            
        Returns:
            str: Formatted log message
        """
        # Get the original formatted message
        formatted = super().format(record)
        
        if self.use_colors and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            # Add color based on level
            levelname = record.levelname
            color = COLORS.get(levelname, COLORS['RESET'])
            reset = COLORS['RESET']
            
            # Replace levelname with colored version
            formatted = formatted.replace(
                f" - {levelname} - ",
                f" - {color}{levelname}{reset} - "
            )
        
        return formatted


# ============================================================================
# LOGGER SETUP
# ============================================================================

def setup_logger(
    name: str = "psiqit",
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    use_colors: bool = True,
    propagate: bool = False
) -> logging.Logger:
    """
    Setup and configure a logger
    
    Args:
        name: Logger name
        level: Log level (string or int)
        log_file: Optional file path for logging to file
        use_colors: Enable colors in terminal output
        propagate: Propagate to parent loggers
        
    Returns:
        logging.Logger: Configured logger
        
    Example:
        >>> logger = setup_logger('psiqit', 'DEBUG')
        >>> logger.info("Hello, world!")
        >>> logger.debug("Debug message")
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.upper(), logging.INFO)
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate
    
    # Remove existing handlers to avoid duplication
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Set formatter
    formatter = ColoredFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        use_colors=use_colors
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "psiqit") -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Logger instance
        
    Example:
        >>> logger = get_logger()
        >>> logger.info("Hello, world!")
    """
    return logging.getLogger(name)


# ============================================================================
# GLOBAL LOGGER INSTANCE
# ============================================================================

# Create global logger instance
_logger: Optional[logging.Logger] = None


def get_global_logger() -> logging.Logger:
    """
    Get the global PSIQIT logger instance (singleton)
    
    Returns:
        logging.Logger: Global logger
        
    Example:
        >>> logger = get_global_logger()
        >>> logger.info("PSIQIT initialized")
    """
    global _logger
    if _logger is None:
        _logger = setup_logger('psiqit', 'INFO')
    return _logger


def set_log_level(level: Union[str, int]):
    """
    Set the log level for the global logger
    
    Args:
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        
    Example:
        >>> set_log_level('DEBUG')
    """
    logger = get_global_logger()
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(level)


def set_log_file(log_file: str):
    """
    Add a file handler to the global logger
    
    Args:
        log_file: Path to log file
        
    Example:
        >>> set_log_file('psiqit.log')
    """
    logger = get_global_logger()
    
    # Check if file handler already exists
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    
    # Add new file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logger.level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def disable_logging():
    """
    Disable all logging by adding a NullHandler
    
    Example:
        >>> disable_logging()
    """
    logger = get_global_logger()
    logger.addHandler(logging.NullHandler())


def enable_logging():
    """
    Enable logging (remove NullHandler)
    
    Example:
        >>> enable_logging()
    """
    logger = get_global_logger()
    for handler in logger.handlers:
        if isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)


# ============================================================================
# LOGGER INTERFACE FUNCTIONS
# ============================================================================

def debug(msg: str, *args, **kwargs):
    """
    Log a debug message
    
    Args:
        msg: Message to log
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Example:
        >>> debug("This is a debug message")
    """
    get_global_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    """
    Log an info message
    
    Args:
        msg: Message to log
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Example:
        >>> info("Quantum circuit created")
    """
    get_global_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """
    Log a warning message
    
    Args:
        msg: Message to log
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Example:
        >>> warning("Potential accuracy loss detected")
    """
    get_global_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """
    Log an error message
    
    Args:
        msg: Message to log
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Example:
        >>> error("Simulation failed")
    """
    get_global_logger().error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs):
    """
    Log a critical message
    
    Args:
        msg: Message to log
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Example:
        >>> critical("Critical error occurred")
    """
    get_global_logger().critical(msg, *args, **kwargs)


def exception(msg: str, *args, **kwargs):
    """
    Log an exception with traceback
    
    Args:
        msg: Message to log
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Example:
        >>> try:
        ...     raise ValueError("Something went wrong")
        ... except Exception as e:
        ...     exception("Exception occurred")
    """
    get_global_logger().exception(msg, *args, **kwargs)


# ============================================================================
# LOGGER DECORATORS
# ============================================================================

def log_function_call(func):
    """
    Decorator to log function calls
    
    Args:
        func: Function to decorate
        
    Returns:
        callable: Decorated function
        
    Example:
        >>> @log_function_call
        ... def my_function(x):
        ...     return x * 2
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_global_logger()
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logger.debug(f"{func.__name__} returned {result}")
        return result
    
    return wrapper


def log_exceptions(func):
    """
    Decorator to log exceptions
    
    Args:
        func: Function to decorate
        
    Returns:
        callable: Decorated function
        
    Example:
        >>> @log_exceptions
        ... def my_function(x):
        ...     return 1 / x
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_global_logger()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in {func.__name__}: {e}")
            raise
    
    return wrapper


def log_time(func):
    """
    Decorator to log execution time
    
    Args:
        func: Function to decorate
        
    Returns:
        callable: Decorated function
        
    Example:
        >>> @log_time
        ... def my_function():
        ...     time.sleep(1)
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_global_logger()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.4f}s")
        return result
    
    return wrapper


# ============================================================================
# EXPORTS
# ============================================================================

# Export the global logger instance
logger = get_global_logger()

# Export functions for convenience
__all__ = [
    # Setup functions
    'setup_logger',
    'get_logger',
    'get_global_logger',
    'set_log_level',
    'set_log_file',
    'disable_logging',
    'enable_logging',
    
    # Logging functions
    'debug',
    'info',
    'warning',
    'error',
    'critical',
    'exception',
    
    # Decorators
    'log_function_call',
    'log_exceptions',
    'log_time',
    
    # The logger instance
    'logger',
]

# ============================================================================
# INITIAL SETUP
# ============================================================================

# Setup default logger
setup_logger('psiqit', 'INFO')