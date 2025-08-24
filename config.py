"""
Main Configuration Module for Recursive Expert Delegation Framework

This module provides a unified configuration interface that integrates both
the basic project settings and the RED framework specific settings.

CONFIGURATION PHILOSOPHY:
This configuration system follows a clear separation of concerns:

1. CONFIG FILE CONTAINS (broad application controls):
   - Algorithm choices (classifier type, embedding models)
   - Behavioral switches (use_embeddings, caching enabled)
   - Infrastructure settings (batch sizes, file paths)
   - Business logic thresholds (confidence thresholds, validation splits)

2. CODE CONTAINS (implementation details):
   - Algorithm-specific hyperparameters (sklearn max_iter, n_estimators)
   - Internal confidence levels for response parsing
   - Retry logic parameters (backoff factors, retry counts)
   - Library defaults that rarely need user modification

This separation ensures that:
- Users can easily configure behavior without needing ML expertise
- Developers can tune algorithms without exposing complexity
- Config files remain clean and focused on user concerns
- Implementation details stay with their respective algorithms

NOTE: This is the single source of truth for all configuration values.
The YAML configuration files are loaded and integrated into this configuration.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any

import dotenv
from pydantic.v1 import BaseSettings

# Load environment variables
dotenv.load_dotenv(Path(__file__).parent / ".env")

class Config(BaseSettings):
    """
    Unified configuration class that consolidates all settings.
    
    This class provides default values for basic LLM settings and loads
    additional RED framework settings from YAML configuration files.
    """

    # -- Default LLM Settings --
    default_model: str = "glm-4.5-air"
    default_temperature: float = 0.0
    default_max_tokens: Optional[int] = 1024
    
    default_timeout: int = 120
    default_max_retries: int = 2
    
    default_system_prompt: str = "you are a helpful assistant"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load RED framework configuration if available
        self._red_config = self._load_red_config()
    
    def _load_red_config(self) -> Dict[str, Any]:
        """
        Load the RED framework configuration from YAML files.
        
        INTEGRATION PATTERN EXPLANATION:
        This method uses a try-catch pattern to conditionally load the RED
        framework configuration. This pattern allows the config to work in
        multiple contexts:
        
        1. When used within the full RED framework (loads YAML config)
        2. When used standalone (gracefully falls back to base config)
        
        This pattern is necessary because:
        - The config module serves both standalone and framework usage
        - It maintains backward compatibility
        - It follows the "graceful degradation" principle
        
        Alternative approaches considered:
        - Factory pattern (too complex for this use case)  
        - Abstract base class (breaks simplicity)
        - Required dependency (breaks standalone usage)
        """
        try:
            from src.red.config.config_loader import get_config as get_red_config
            return get_red_config()
        except (ImportError, FileNotFoundError):
            # Return empty dict if RED config is not available
            return {}
    
    def get_red_config(self) -> Dict[str, Any]:
        """Get the RED framework configuration."""
        return self._red_config
    
    def get(self, key: str, default=None):
        """Get a configuration value, checking RED config first, then base config."""
        # First check if it's in the RED configuration
        if key in self._red_config:
            return self._red_config[key]
        
        # Then check if it's a base config attribute
        if hasattr(self, key):
            return getattr(self, key)
        
        return default


@lru_cache
def get_config() -> Config:
    """Get the unified configuration instance."""
    return Config()
