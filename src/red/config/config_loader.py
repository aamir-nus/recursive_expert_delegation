"""
Configuration Loader for R.E.D. Framework

This module provides utilities for loading and managing configuration files
for the R.E.D. framework.

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
"""

import yaml
import dotenv
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic.v1 import BaseSettings

# Load environment variables from project root
project_root = Path(__file__).resolve().parents[3]
dotenv.load_dotenv(project_root / ".env")

class LLMConfig(BaseSettings):
    """
    Basic LLM configuration settings loaded from environment variables.
    """
    
    # -- Default LLM Settings --
    default_model: str = "glm-4.5-air"
    default_temperature: float = 0.0
    default_max_tokens: Optional[int] = 1024
    
    default_timeout: int = 120
    default_max_retries: int = 2
    
    default_system_prompt: str = "you are a helpful assistant"


class Config:
    """
    Unified configuration class that consolidates all settings.
    
    This class provides default values for basic LLM settings and loads
    additional RED framework settings from YAML configuration files.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        # Load basic LLM configuration from environment
        self._llm_config = LLMConfig()
        
        # Load RED framework configuration from YAML
        self._config_loader = ConfigLoader(config_dir)
        self._red_config = self._config_loader.load_main_config()
    
    # Expose LLM config attributes directly
    @property
    def default_model(self) -> str:
        return self._llm_config.default_model
    
    @property
    def default_temperature(self) -> float:
        return self._llm_config.default_temperature
    
    @property
    def default_max_tokens(self) -> Optional[int]:
        return self._llm_config.default_max_tokens
    
    @property
    def default_timeout(self) -> int:
        return self._llm_config.default_timeout
    
    @property
    def default_max_retries(self) -> int:
        return self._llm_config.default_max_retries
    
    @property
    def default_system_prompt(self) -> str:
        return self._llm_config.default_system_prompt
    
    def get(self, key: str, default=None):
        """Get a configuration value, checking RED config first, then base config."""
        # Get resolved configuration (with auto values resolved)
        resolved_config = self.resolve_performance_config()
        
        # First check if it's in the resolved RED configuration
        if key in resolved_config:
            return resolved_config[key]
        
        # Then check if it's a base config attribute
        if hasattr(self, key):
            return getattr(self, key)
        
        return default
    
    def get_red_config(self) -> Dict[str, Any]:
        """Get the RED framework configuration."""
        return self._red_config
    
    def get_prompt(self, prompt_name: str, category: str = None) -> Optional[str]:
        """Get a prompt template from the prompts configuration."""
        return self._config_loader.get_prompt(prompt_name, category)
    
    def resolve_performance_config(self) -> Dict[str, Any]:
        """
        Resolve auto configurations based on performance mode.
        
        Returns:
            Resolved configuration with auto values replaced
        """
        config = self._red_config.copy()
        performance_mode = config.get('performance', {}).get('mode', 'balanced')
        
        # Resolve embedding model
        embeddings = config.get('embeddings', {})
        if embeddings.get('model_name') == 'auto':
            model_key = f"{performance_mode}_model"
            resolved_model = embeddings.get(model_key, embeddings.get('balanced_model', 'all-mpnet-base-v2'))
            config['embeddings']['model_name'] = resolved_model
        
        # Resolve classifier type
        classifier = config.get('classifier', {})
        if classifier.get('type') == 'auto':
            type_key = f"{performance_mode}_type"
            resolved_type = classifier.get(type_key, classifier.get('balanced_type', 'random_forest'))
            config['classifier']['type'] = resolved_type
        
        # Resolve UMAP components
        subsetting = config.get('subsetting', {})
        if subsetting.get('umap_components') == 'auto':
            components_key = f"{performance_mode}_components"
            resolved_components = subsetting.get(components_key, subsetting.get('balanced_components', 50))
            config['subsetting']['umap_components'] = resolved_components
        
        return config

class ConfigLoader:
    """
    Loads and manages configuration files for the R.E.D. framework.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the config loader.
        
        Args:
            config_dir: Directory containing config files. If None, uses default.
        """
        if config_dir is None:
            self.config_dir = Path(__file__).parent
        else:
            self.config_dir = Path(config_dir)
        
        self._main_config = None
        self._prompts_config = None
    
    def load_main_config(self) -> Dict[str, Any]:
        """
        Load the main configuration file.
        
        Returns:
            Main configuration dictionary
        """
        if self._main_config is None:
            config_path = self.config_dir / "main_config.yaml"
            
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._main_config = yaml.safe_load(f)
            except FileNotFoundError:
                print(f"Main config file not found at {config_path}")
                self._main_config = self._get_default_main_config()
            except yaml.YAMLError as e:
                print(f"Error parsing main config: {e}")
                self._main_config = self._get_default_main_config()
        
        return self._main_config
    
    def load_prompts_config(self) -> Dict[str, Any]:
        """
        Load the prompts configuration file.
        
        Returns:
            Prompts configuration dictionary
        """
        if self._prompts_config is None:
            prompts_path = self.config_dir / "prompts.yaml"
            
            try:
                with open(prompts_path, 'r', encoding='utf-8') as f:
                    self._prompts_config = yaml.safe_load(f)
            except FileNotFoundError:
                print(f"Prompts config file not found at {prompts_path}")
                self._prompts_config = self._get_default_prompts_config()
            except yaml.YAMLError as e:
                print(f"Error parsing prompts config: {e}")
                self._prompts_config = self._get_default_prompts_config()
        
        return self._prompts_config
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the config value (e.g., 'data.data_dir')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        config = self.load_main_config()
        
        keys = key_path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def get_prompt_template(self, template_name: str, category: str = None) -> str:
        """
        Get a prompt template by name.
        
        Args:
            template_name: Name of the template
            category: Category of the template (e.g., 'validation_prompts')
            
        Returns:
            Prompt template string
        """
        prompts_config = self.load_prompts_config()
        
        if category:
            templates = prompts_config.get(category, {})
            template = templates.get(template_name, "")
        else:
            # Search all categories
            template = ""
            for cat_name, templates in prompts_config.items():
                if isinstance(templates, dict) and template_name in templates:
                    template = templates[template_name]
                    break
        
        # If template not found, try fallback defaults
        if not template:
            defaults = prompts_config.get('defaults', {})
            if template_name == 'validation_expert':
                template = defaults.get('fallback_system_prompt', "")
            elif template_name == 'binary_validation':
                template = defaults.get('fallback_validation_prompt', "")
        
        return template
    
    def get_system_prompt(self, prompt_name: str) -> str:
        """
        Get a system prompt by name.
        
        Args:
            prompt_name: Name of the system prompt
            
        Returns:
            System prompt string
        """
        return self.get_prompt_template(prompt_name, 'system_prompts')
    
    def get_validation_prompt(self, prompt_name: str) -> str:
        """
        Get a validation prompt template by name.
        
        Args:
            prompt_name: Name of the validation prompt
            
        Returns:
            Validation prompt template string
        """
        return self.get_prompt_template(prompt_name, 'validation_prompts')
    
    def _get_default_main_config(self) -> Dict[str, Any]:
        """Get default main configuration."""
        return {
            'data': {
                'data_dir': './data',
                'output_dir': './outputs',
                'log_dir': './logs',
                'encoding': 'utf-8',
                'text_column': 'text',
                'label_column': 'label'
            },
            'subsetting': {
                'subset_size': 8,
                'use_umap': True,
                'umap_components': 50,
                'random_state': 42
            },
            'embeddings': {
                'model_name': 'all-MiniLM-L6-v2',
                'enable_cache': True,
                'cache_dir': 'embeddings_cache',
                'device': 'auto'
            },
            'classifier': {
                'type': 'logistic_regression',
                'use_embeddings': True,
                'noise_oversample_factor': 2.0,
                'max_features': 10000,
                'validation_split': 0.2,
                'random_state': 42
            },
            'llm_validation': {
                'model_name': 'deepseek-r1-8b',
                'temperature': 0.0,
                'max_timeout': 30,
                'similar_examples_count': 3,
                'confidence_threshold': 0.5,
                'use_cache': True,
                'auto_generate_descriptions': True
            },
            'active_learning': {
                'batch_size': 100,
                'samples_per_iteration': 50,
                'max_iterations': 10,
                'min_information_gain': 0.01,
                'retrain_threshold': 100,
                'retrain_all': False
            }
        }
    
    def _get_default_prompts_config(self) -> Dict[str, Any]:
        """Get default prompts configuration."""
        return {
            'system_prompts': {
                'validation_expert': """You are an expert text classifier performing validation tasks.
Your role is to carefully evaluate whether text samples truly belong to their predicted class labels.
Be thorough but decisive in your analysis. Respond with only 'True' or 'False'."""
            },
            'validation_prompts': {
                'binary_validation': """Class: {label}
Description: {description}

{examples_section}

Text: "{text}"

Does this text belong to the "{label}" class?
Respond with ONLY "True" or "False"."""
            }
        }
    
    def reload_configs(self) -> None:
        """Reload all configuration files."""
        self._main_config = None
        self._prompts_config = None
        print("Configuration files reloaded")
    
    def save_config(self, config: Dict[str, Any], filename: str) -> None:
        """
        Save a configuration dictionary to a YAML file.
        
        Args:
            config: Configuration dictionary to save
            filename: Name of the file to save (with .yaml extension)
        """
        filepath = self.config_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            print(f"Configuration saved to {filepath}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

# Global config loader instance
_config_loader = None

def get_config_loader(config_dir: Optional[str] = None) -> ConfigLoader:
    """
    Get the global config loader instance.
    
    Args:
        config_dir: Directory containing config files
        
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None or config_dir is not None:
        _config_loader = ConfigLoader(config_dir)
    return _config_loader

# Global config instance
_config_instance = None

def get_config() -> Config:
    """Get the unified configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

def get_config_value(key_path: str = None, default: Any = None) -> Any:
    """
    Get configuration value using the global config loader.
    
    Args:
        key_path: Dot-separated path to config value, or None for full config
        default: Default value if key not found
        
    Returns:
        Configuration value or full config dictionary
    """
    loader = get_config_loader()
    
    if key_path is None:
        return loader.load_main_config()
    else:
        return loader.get_config_value(key_path, default)

def get_prompt(template_name: str, category: str = None) -> str:
    """
    Get prompt template using the global config loader.
    
    Args:
        template_name: Name of the template
        category: Category of the template
        
    Returns:
        Prompt template string
    """
    loader = get_config_loader()
    return loader.get_prompt_template(template_name, category)
