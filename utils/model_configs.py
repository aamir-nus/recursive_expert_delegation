from typing import Dict, Any, List, Optional
import os
from dataclasses import dataclass
from enum import Enum

class ModelFamily(Enum):
    CLAUDE = "claude"
    GEMINI = "gemini"
    PERPLEXITY = "perplexity"

@dataclass
class ModelConfig:
    base_url: str
    api_key: str
    headers: Dict[str, str]
    json_params: Dict[str, Any]
    model_name: str
    message_formatter: Any  # Function to format messages for this model

class ModelConfigFactory:
    @staticmethod
    def get_claude_config(model_name: str) -> ModelConfig:
        def format_claude_messages(messages: List[Dict[str, str]]) -> Dict[str, Any]:
            # Claude expects messages in a specific format
            formatted_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    # For Claude, system messages are part of the user message
                    formatted_messages.append({
                        "role": "user",
                        "content": f"System: {msg['content']}"
                    })
                else:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            return {
                "model": model_name,
                "max_tokens": 1024,
                "messages": formatted_messages
            }
        
        return ModelConfig(
            base_url="https://api.anthropic.com/v1/messages",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            headers={
                "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json_params={},  # All parameters will be handled by the formatter
            model_name=model_name,
            message_formatter=format_claude_messages
        )

    @staticmethod
    def get_gemini_config(model_name: str) -> ModelConfig:
        base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        endpoint = f"{base_url}/{model_name}:generateContent?key={api_key}"
        
        def format_gemini_messages(messages: List[Dict[str, str]]) -> Dict[str, Any]:
            formatted_messages = []
            
            # Start with system prompt as user message
            system_prompt = next((msg for msg in messages if msg["role"] == "system"), None)
            if system_prompt:
                formatted_messages.append({
                    "role": "user",
                    "parts": [{"text": system_prompt["content"]}]
                })
                formatted_messages.append({
                    "role": "model",
                    "parts": [{"text": "Understood. Continue."}]
                })
            
            # Add few-shot examples if any
            for msg in messages:
                if msg["role"] in ["user", "assistant"]:
                    formatted_messages.append({
                        "role": "user" if msg["role"] == "user" else "model",
                        "parts": [{"text": msg["content"]}]
                    })
            
            return {
                "contents": formatted_messages,
                "safety_settings": [
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ],
                "generationConfig": {"temperature": 0.0}
            }
        
        return ModelConfig(
            base_url=endpoint,
            api_key=api_key,
            headers={"Content-Type": "application/json"},
            json_params={},
            model_name=model_name,
            message_formatter=format_gemini_messages
        )

    @staticmethod
    def get_perplexity_config(model_name: str) -> ModelConfig:
        return ModelConfig(
            base_url="https://api.perplexity.ai/chat/completions",
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            headers={
                "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
                "Content-Type": "application/json"
            },
            json_params={
                "max_tokens": 300,
                "temperature": 0.0,
                "top_p": 0.9,
                "search_domain_filter": ["youtube.com", "medium.com", "reddit.com", "stackoverflow.com", "github.com", "google.com"],
                "return_images": False,
                "return_related_questions": False,
                "search_recency_filter": "month",
                "top_k": 0,
                "stream": False,
                "presence_penalty": 0,
                "frequency_penalty": 1,
                "response_format": {},
                "web_search_options": {"search_context_size": "high"}
            },
            model_name=model_name,
            message_formatter=lambda messages: {
                "messages": messages,
                "model": model_name,
                "max_tokens": 300,
                "temperature": 0.0
            }
        )

def get_model_configs(model_name: str) -> ModelConfig:
    """
    Returns the appropriate API configurations based on the model name.
    
    Args:
        model_name (str): Name of the model to get configurations for
        
    Returns:
        ModelConfig: Configuration object containing all necessary parameters for the model
    """
    model_mapping = {
        # Claude models
        "claude-3.5-sonnet": (ModelFamily.CLAUDE, "claude-3-5-sonnet-20241022"),
        "claude-3.7-sonnet": (ModelFamily.CLAUDE, "claude-3-7-sonnet-20241022"),
        
        # Gemini models
        "gemini-2.0-flash": (ModelFamily.GEMINI, "gemini-2.0-flash"),
        "gemini-2.0-flash-thinking-exp": (ModelFamily.GEMINI, "gemini-2.0-flash-thinking-exp"),
        "gemini-2.5-flash": (ModelFamily.GEMINI, "gemini-2.5-flash"),
        "gemini-2.5-pro-exp": (ModelFamily.GEMINI, "gemini-2.5-pro-preview-03-25"),

        # Perplexity models
        "sonar": (ModelFamily.PERPLEXITY, "sonar"),
        "sonar-small": (ModelFamily.PERPLEXITY, "sonar-small")
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(model_mapping.keys())}")
    
    family, actual_model_name = model_mapping[model_name]
    
    if family == ModelFamily.CLAUDE:
        config = ModelConfigFactory.get_claude_config(actual_model_name)
    elif family == ModelFamily.GEMINI:
        config = ModelConfigFactory.get_gemini_config(actual_model_name)
    elif family == ModelFamily.PERPLEXITY:
        config = ModelConfigFactory.get_perplexity_config(actual_model_name)
    else:
        raise ValueError(f"Unsupported model family: {family}")
    
    return config