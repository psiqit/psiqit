# psiqit/utils/config.py

"""
Configuration Module
Global configuration settings for PSIQIT
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from ..utils.logger import logger


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """
    Global configuration for PSIQIT
    
    Attributes:
        debug: Enable debug mode
        verbose: Enable verbose output
        seed: Global random seed (None for random)
        hbar: Reduced Planck constant (natural units)
        default_shots: Default number of measurement shots
        default_precision: Default numerical precision
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        cache_enabled: Enable caching of results
        cache_size: Maximum cache size
        tensor_network: Tensor network backend ('numpy', 'opt_einsum')
        parallel: Enable parallel processing
        num_threads: Number of threads for parallel processing
        save_path: Default save path for results
        load_path: Default load path for data
    """
    # General
    debug: bool = False
    verbose: bool = True
    seed: Optional[int] = None
    
    # Physics
    hbar: float = 1.0
    
    # Default values
    default_shots: int = 1024
    default_precision: int = 6
    
    # Logging
    log_level: str = "INFO"
    
    # Performance
    cache_enabled: bool = True
    cache_size: int = 100
    tensor_network: str = "numpy"
    parallel: bool = False
    num_threads: int = 4
    
    # Paths
    save_path: str = "./results"
    load_path: str = "./data"
    
    # Advanced
    use_advanced_optimization: bool = False
    optimization_tolerance: float = 1e-6
    max_iterations: int = 1000
    
    def __post_init__(self):
        """Post-process configuration"""
        # Ensure paths exist
        if not os.path.exists(self.save_path):
            try:
                os.makedirs(self.save_path, exist_ok=True)
            except:
                pass
        
        # Set logging level
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        import logging
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
        }
        level = level_map.get(self.log_level.upper(), logging.INFO)
        logger.setLevel(level)
    
    def update(self, **kwargs):
        """
        Update configuration with keyword arguments
        
        Args:
            **kwargs: Configuration key-value pairs
            
        Example:
            >>> config.update(debug=True, default_shots=2048)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Config updated: {key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key}")
        
        # Re-setup logging if log_level changed
        if 'log_level' in kwargs:
            self._setup_logging()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            'debug': self.debug,
            'verbose': self.verbose,
            'seed': self.seed,
            'hbar': self.hbar,
            'default_shots': self.default_shots,
            'default_precision': self.default_precision,
            'log_level': self.log_level,
            'cache_enabled': self.cache_enabled,
            'cache_size': self.cache_size,
            'tensor_network': self.tensor_network,
            'parallel': self.parallel,
            'num_threads': self.num_threads,
            'save_path': self.save_path,
            'load_path': self.load_path,
            'use_advanced_optimization': self.use_advanced_optimization,
            'optimization_tolerance': self.optimization_tolerance,
            'max_iterations': self.max_iterations,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert configuration to JSON string
        
        Args:
            indent: JSON indentation
            
        Returns:
            str: JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: Optional[str] = None):
        """
        Save configuration to file
        
        Args:
            path: File path (if None, uses default)
            
        Example:
            >>> config.save('my_config.json')
        """
        if path is None:
            path = os.path.join(self.save_path, 'config.json')
        
        with open(path, 'w') as f:
            f.write(self.to_json())
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """
        Load configuration from file
        
        Args:
            path: File path
            
        Returns:
            Config: Loaded configuration
            
        Example:
            >>> config = Config.load('my_config.json')
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        logger.info(f"Configuration loaded from {path}")
        return config
    
    def __repr__(self) -> str:
        return f"Config(debug={self.debug}, verbose={self.verbose}, seed={self.seed})"
    
    def __str__(self) -> str:
        lines = [
            "PSIQIT Configuration:",
            f"  Debug: {self.debug}",
            f"  Verbose: {self.verbose}",
            f"  Seed: {self.seed}",
            f"  hbar: {self.hbar}",
            f"  Default Shots: {self.default_shots}",
            f"  Default Precision: {self.default_precision}",
            f"  Log Level: {self.log_level}",
            f"  Cache Enabled: {self.cache_enabled}",
            f"  Cache Size: {self.cache_size}",
            f"  Tensor Network: {self.tensor_network}",
            f"  Parallel: {self.parallel}",
            f"  Num Threads: {self.num_threads}",
            f"  Save Path: {self.save_path}",
            f"  Load Path: {self.load_path}",
        ]
        return "\n".join(lines)


# ============================================================================
# SINGLETON CONFIG INSTANCE
# ============================================================================

_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance (singleton)
    
    Returns:
        Config: Global configuration
        
    Example:
        >>> config = get_config()
        >>> config.default_shots = 2048
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config):
    """
    Set the global configuration instance
    
    Args:
        config: Configuration instance
        
    Example:
        >>> config = Config()
        >>> config.debug = True
        >>> set_config(config)
    """
    global _config
    _config = config
    logger.info("Global configuration updated")


def reset_config():
    """
    Reset the global configuration to default
    
    Example:
        >>> reset_config()
    """
    global _config
    _config = Config()
    logger.info("Configuration reset to default")


# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

def load_from_env():
    """
    Load configuration from environment variables
    
    Environment variables:
        PSIQIT_DEBUG: Set debug mode (true/false)
        PSIQIT_VERBOSE: Set verbose mode (true/false)
        PSIQIT_SEED: Set random seed (integer)
        PSIQIT_HBAR: Set hbar value (float)
        PSIQIT_DEFAULT_SHOTS: Set default shots (integer)
        PSIQIT_LOG_LEVEL: Set log level (DEBUG, INFO, WARNING, ERROR)
        PSIQIT_SAVE_PATH: Set save path (string)
        PSIQIT_LOAD_PATH: Set load path (string)
        PSIQIT_NUM_THREADS: Set number of threads (integer)
    
    Returns:
        Config: Updated configuration
        
    Example:
        >>> config = load_from_env()
    """
    config = get_config()
    
    # Boolean values
    if os.getenv('PSIQIT_DEBUG', '').lower() in ('true', '1', 'yes'):
        config.debug = True
    if os.getenv('PSIQIT_VERBOSE', '').lower() in ('false', '0', 'no'):
        config.verbose = False
    
    # Integer values
    if os.getenv('PSIQIT_SEED'):
        config.seed = int(os.getenv('PSIQIT_SEED'))
    if os.getenv('PSIQIT_DEFAULT_SHOTS'):
        config.default_shots = int(os.getenv('PSIQIT_DEFAULT_SHOTS'))
    if os.getenv('PSIQIT_NUM_THREADS'):
        config.num_threads = int(os.getenv('PSIQIT_NUM_THREADS'))
    
    # Float values
    if os.getenv('PSIQIT_HBAR'):
        config.hbar = float(os.getenv('PSIQIT_HBAR'))
    
    # String values
    if os.getenv('PSIQIT_LOG_LEVEL'):
        config.log_level = os.getenv('PSIQIT_LOG_LEVEL')
    if os.getenv('PSIQIT_SAVE_PATH'):
        config.save_path = os.getenv('PSIQIT_SAVE_PATH')
    if os.getenv('PSIQIT_LOAD_PATH'):
        config.load_path = os.getenv('PSIQIT_LOAD_PATH')
    
    # Re-setup logging
    config._setup_logging()
    
    logger.info("Configuration loaded from environment variables")
    return config


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'Config',
    'get_config',
    'set_config',
    'reset_config',
    'load_from_env',
]