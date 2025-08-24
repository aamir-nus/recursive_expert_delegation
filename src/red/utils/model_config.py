from typing import Dict, List, Any
from enum import Enum
import sys
import os

# Import existing model configurations
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from utils.model_configs import ModelFamily as ExistingModelFamily, get_model_configs
from ..config.config_loader import get_prompt

# Re-export the existing ModelFamily for compatibility
ModelFamily = ExistingModelFamily

class PromptFormatter:
    """
    Handles formatting of prompts for different model families.
    
    This class provides a unified interface for formatting system and user prompts
    according to the specific requirements of different LLM APIs.
    """
    
    @staticmethod
    def format_prompt(system_prompt: str, 
                     user_prompt: str, 
                     model_family: ModelFamily,
                     few_shot_examples: List[List[str]] = None) -> Dict[str, Any]:
        """
        Format a prompt according to the model family requirements.
        
        Args:
            system_prompt: System instruction
            user_prompt: User query
            model_family: The target model family
            few_shot_examples: List of [user, assistant] example pairs
            
        Returns:
            Formatted prompt dictionary
        """
        if few_shot_examples is None:
            few_shot_examples = []
        
        if model_family == ModelFamily.OPENROUTER or model_family == ModelFamily.OLLAMA:
            return PromptFormatter._format_openai_style(
                system_prompt, user_prompt, few_shot_examples
            )
        elif model_family == ModelFamily.CLAUDE:
            return PromptFormatter._format_claude_style(
                system_prompt, user_prompt, few_shot_examples
            )
        elif model_family == ModelFamily.GEMINI:
            return PromptFormatter._format_gemini_style(
                system_prompt, user_prompt, few_shot_examples
            )
        elif model_family == ModelFamily.PERPLEXITY:
            return PromptFormatter._format_openai_style(
                system_prompt, user_prompt, few_shot_examples
            )
        else:
            # Default to OpenAI style for other models
            return PromptFormatter._format_openai_style(
                system_prompt, user_prompt, few_shot_examples
            )
    
    @staticmethod
    def _format_openai_style(system_prompt: str, 
                           user_prompt: str, 
                           few_shot_examples: List[List[str]]) -> Dict[str, Any]:
        """Format prompt for OpenAI-style APIs (including Ollama, OpenRouter)."""
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add few-shot examples
        for example in few_shot_examples:
            if len(example) >= 2:
                messages.append({"role": "user", "content": example[0]})
                messages.append({"role": "assistant", "content": example[1]})
        
        # Add user prompt
        messages.append({"role": "user", "content": user_prompt})
        
        return {"messages": messages}
    
    @staticmethod
    def _format_claude_style(system_prompt: str, 
                           user_prompt: str, 
                           few_shot_examples: List[List[str]]) -> Dict[str, Any]:
        """Format prompt for Claude API."""
        messages = []
        
        # For Claude, system message is often incorporated into the user message
        # or handled separately in the API call
        content_parts = []
        
        if system_prompt:
            content_parts.append(f"System: {system_prompt}")
        
        # Add few-shot examples
        for example in few_shot_examples:
            if len(example) >= 2:
                content_parts.append(f"Human: {example[0]}")
                content_parts.append(f"Assistant: {example[1]}")
        
        # Add user prompt
        content_parts.append(f"Human: {user_prompt}")
        
        messages.append({
            "role": "user",
            "content": "\n\n".join(content_parts)
        })
        
        return {"messages": messages}
    
    @staticmethod
    def _format_gemini_style(system_prompt: str, 
                           user_prompt: str, 
                           few_shot_examples: List[List[str]]) -> Dict[str, Any]:
        """Format prompt for Gemini API."""
        contents = []
        
        # Start with system prompt as user message if present
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": system_prompt}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Understood. I'll follow these instructions."}]
            })
        
        # Add few-shot examples
        for example in few_shot_examples:
            if len(example) >= 2:
                contents.append({
                    "role": "user",
                    "parts": [{"text": example[0]}]
                })
                contents.append({
                    "role": "model",
                    "parts": [{"text": example[1]}]
                })
        
        # Add user prompt
        contents.append({
            "role": "user",
            "parts": [{"text": user_prompt}]
        })
        
        return {"contents": contents}
    
    @staticmethod
    def detect_model_family(model_name: str) -> ModelFamily:
        """
        Detect the model family based on the model name.
        
        Uses the existing model configurations to determine family.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Detected model family
        """
        try:
            # Try to get model config to determine family
            model_config = get_model_configs(model_name)
            
            # Check the base URL to determine family
            base_url = model_config.base_url.lower()
            
            if "anthropic" in base_url or "claude" in base_url:
                return ModelFamily.CLAUDE
            elif "generativelanguage.googleapis.com" in base_url or "gemini" in base_url:
                return ModelFamily.GEMINI
            elif "perplexity" in base_url:
                return ModelFamily.PERPLEXITY
            elif "openrouter" in base_url:
                return ModelFamily.OPENROUTER
            elif "localhost" in base_url or "ollama" in base_url:
                return ModelFamily.OLLAMA
            else:
                return ModelFamily.OPENROUTER  # Default fallback
                
        except Exception:
            # Fallback to name-based detection
            model_name_lower = model_name.lower()
            
            if "claude" in model_name_lower:
                return ModelFamily.CLAUDE
            elif "gemini" in model_name_lower:
                return ModelFamily.GEMINI
            elif "perplexity" in model_name_lower or "sonar" in model_name_lower:
                return ModelFamily.PERPLEXITY
            elif "deepseek" in model_name_lower or "qwen" in model_name_lower or "phi" in model_name_lower:
                return ModelFamily.OLLAMA
            else:
                return ModelFamily.OPENROUTER

class ValidationPrompts:
    """
    Pre-defined prompt templates for different validation tasks.
    
    All prompts are now loaded from configuration files to avoid magic values.
    """
    
    @staticmethod
    def get_binary_validation_prompt() -> str:
        """Get binary classification validation prompt from config."""
        template = get_prompt('binary_validation', 'validation_prompts')
        if not template:
            # Fallback template if not in config
            template = """Class: {label}
Description: {description}

{examples_section}

Text: "{text}"

Does this text belong to the "{label}" class?
Respond with ONLY "True" or "False"."""
        return template
    
    @staticmethod
    def get_multi_class_validation_prompt() -> str:
        """Get multi-class validation prompt from config."""
        template = get_prompt('multiclass_validation', 'validation_prompts')
        if not template:
            # Fallback template if not in config
            template = """Available Classes:
{class_descriptions}

Text: "{text}"
Predicted Class: {predicted_class}

{examples_section}

Is the predicted class correct?
Respond with ONLY "True" or "False"."""
        return template
    
    @staticmethod
    def get_uncertainty_assessment_prompt() -> str:
        """Get uncertainty assessment prompt from config."""
        template = get_prompt('assess_uncertainty', 'uncertainty_prompts')
        if not template:
            # Fallback template if not in config
            template = """Text: "{text}"
Predicted Class: {predicted_class}

Rate the uncertainty level:
CERTAIN, LIKELY, UNCERTAIN, UNLIKELY

Respond with ONLY one of these levels."""
        return template
    
    @staticmethod
    def format_validation_prompt(template: str, **kwargs) -> str:
        """
        Format a validation prompt template with provided arguments.
        
        Args:
            template: The prompt template string
            **kwargs: Template formatting arguments
            
        Returns:
            Formatted prompt string
        """
        return template.format(**kwargs)
    
    @staticmethod
    def build_examples_section(examples: List[str], max_examples: int = None) -> str:
        """
        Build the examples section for prompt templates.
        
        Args:
            examples: List of example texts
            max_examples: Maximum number of examples to include (from config if None)
            
        Returns:
            Formatted examples section
        """
        if not examples:
            return ""
        
        # Get max_examples from config if not provided
        if max_examples is None:
            from ..config.config_loader import get_config
            config = get_config()
            max_examples = config.get('llm_validation', {}).get('similar_examples_count', 5)
        
        examples_text = "Representative Examples:\n"
        for i, example in enumerate(examples[:max_examples], 1):
            examples_text += f"{i}. \"{example}\"\n"
        
        return examples_text + "\n"