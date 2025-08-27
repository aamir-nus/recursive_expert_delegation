from typing import List, Dict, Any
from dataclasses import dataclass

# Import existing LLM pipeline and model configs
# ANTI-PATTERN EXPLANATION: Path manipulation for cross-module imports
# 
# This sys.path manipulation is necessary because the LLM utilities exist 
# outside the RED framework package structure. While generally avoided,
# this is a controlled use case where:
# 1. The project has legacy utilities in the root that need to be reused
# 2. The path manipulation is defensive (checks if already present)
# 3. It's isolated to this specific module

import sys
from pathlib import Path

# Add the project root to sys.path for importing utils
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.llm_pipeline import LLM
from utils.model_configs import get_model_configs
from ..config.config_loader import get_config, get_prompt

@dataclass
class LLMResponse:
    """Structured response from LLM."""
    content: str
    is_valid: bool
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class LLMClient:
    """
    Unified LLM client for the R.E.D. framework.
    
    This class uses the existing LLM pipeline and removes all hardcoded values,
    relying entirely on configuration files for defaults.
    """
    
    def __init__(self, 
                 model_name: str = None,
                 temperature: float = None,
                 max_timeout: int = None):
        """
        Initialize the LLM client.
        
        Args:
            model_name: Name of the model to use (from config if None)
            temperature: Sampling temperature (from config if None)
            max_timeout: Maximum timeout per request in seconds (from config if None)
        """
        # Load configuration
        config = get_config()
        
        # Set parameters from config or provided values
        self.model_name = model_name or config.get('llm_validation', {}).get('model_name', config.default_model)
        self.temperature = temperature if temperature is not None else config.get('llm_validation', {}).get('temperature', config.default_temperature)
        self.max_timeout = max_timeout or config.get('llm_validation', {}).get('max_timeout', config.default_timeout)
        
        # Get model configurations
        self.model_configs = get_model_configs(self.model_name)
        
    def _create_llm_instance(self, system_prompt: str, few_shot_examples: List[List[str]] = None) -> LLM:
        """Create an instance of the existing LLM class."""
        if few_shot_examples is None:
            few_shot_examples = []
        
        # Use the existing LLM class with our parameters
        return LLM(
            system_prompt=system_prompt,
            few_shot_examples=few_shot_examples,
            model=self.model_name,
            max_timeout_per_request=self.max_timeout
        )
    
    def query(self, 
              system_prompt: str, 
              user_prompt: str, 
              few_shot_examples: List[List[str]] = None) -> LLMResponse:
        """
        Query the LLM with a single prompt.
        
        Args:
            system_prompt: System instruction for the LLM
            user_prompt: User query/prompt
            few_shot_examples: List of [user, assistant] example pairs
            
        Returns:
            LLMResponse object containing the result
        """
        try:
            with self._create_llm_instance(system_prompt, few_shot_examples) as llm:
                # Use the existing batch_generate method with a single prompt
                responses = llm.batch_generate([user_prompt])
                
                if responses and len(responses) > 0:
                    response_content = responses[0]
                    
                    # Check if response is valid (not 'indeterminate')
                    is_valid = response_content != 'indeterminate'
                    
                    return LLMResponse(
                        content=response_content,
                        is_valid=is_valid,
                        metadata={
                            'model': self.model_name,
                            'temperature': self.temperature
                        }
                    )
                else:
                    return LLMResponse(
                        content="",
                        is_valid=False,
                        metadata={'error': 'Empty response'}
                    )
                    
        except Exception as e:
            return LLMResponse(
                content="",
                is_valid=False,
                metadata={'error': str(e)}
            )
    
    def query_batch(self, 
                   system_prompt: str, 
                   user_prompts: List[str], 
                   few_shot_examples: List[List[str]] = None) -> List[LLMResponse]:
        """
        Query the LLM with multiple prompts in batch.
        
        Args:
            system_prompt: System instruction for the LLM
            user_prompts: List of user queries/prompts
            few_shot_examples: List of [user, assistant] example pairs
            
        Returns:
            List of LLMResponse objects
        """
        try:
            with self._create_llm_instance(system_prompt, few_shot_examples) as llm:
                responses = llm.batch_generate(user_prompts)
                
                llm_responses = []
                for i, response_content in enumerate(responses):
                    is_valid = response_content != 'indeterminate'
                    
                    llm_responses.append(LLMResponse(
                        content=response_content,
                        is_valid=is_valid,
                        metadata={
                            'model': self.model_name,
                            'temperature': self.temperature,
                            'batch_index': i
                        }
                    ))
                
                return llm_responses
                
        except Exception as e:
            # Return error responses for all prompts
            return [LLMResponse(
                content="",
                is_valid=False,
                metadata={'error': str(e), 'batch_index': i}
            ) for i in range(len(user_prompts))]
    
    def validate_classification(self, 
                              text: str, 
                              predicted_label: str, 
                              label_description: str, 
                              similar_examples: List[str] = None,
                              validation_prompt_template: str = None) -> bool:
        """
        Validate if a text sample belongs to the predicted label.
        
        Args:
            text: The text sample to validate
            predicted_label: The predicted class label
            label_description: Description of what the label means
            similar_examples: List of similar examples from training data
            validation_prompt_template: Custom prompt template (from config if None)
            
        Returns:
            Boolean indicating if the classification is valid
        """
        if similar_examples is None:
            similar_examples = []
        
        # Get validation prompt template from config if not provided
        if validation_prompt_template is None:
            validation_prompt_template = get_prompt('binary_validation', 'validation_prompts')
            
            # Fallback to a basic template if not found in config
            if not validation_prompt_template:
                validation_prompt_template = """Class: {label}
Description: {description}

{examples_section}

Text: "{text}"

Does this text belong to the "{label}" class?
Respond with ONLY "True" or "False"."""

        # Build examples section
        examples_section = ""
        if similar_examples:
            max_examples = self.red_config.get('llm_validation', {}).get('similar_examples_count', 3)
            examples_section = "Similar Examples from Training Data:\n"
            for i, example in enumerate(similar_examples[:max_examples], 1):
                examples_section += f"{i}. \"{example}\"\n"
            examples_section += "\n"
        
        # Format the prompt
        user_prompt = validation_prompt_template.format(
            label=predicted_label,
            description=label_description,
            examples_section=examples_section,
            text=text
        )
        
        # Get system prompt from config
        system_prompt = get_prompt('validation_expert', 'system_prompts')
        if not system_prompt:
            system_prompt = "You are a precise and accurate text classification validator. Follow instructions exactly."
        
        response = self.query(system_prompt, user_prompt)
        
        if not response.is_valid:
            return False
            
        # Parse the response
        response_text = response.content.strip().lower()
        
        # Look for true/false indicators
        if "true" in response_text and "false" not in response_text:
            return True
        elif "false" in response_text and "true" not in response_text:
            return False
        else:
            # If ambiguous, default to False for safety
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_timeout': self.max_timeout,
            'base_url': self.model_configs.base_url,
            'model_family': self.model_configs.model_name
        }