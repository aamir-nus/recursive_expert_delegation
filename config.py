import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import dotenv
from pydantic.v1 import BaseSettings

dotenv.load_dotenv(Path(__file__).parent.parent.parent / ".env")


class Config(BaseSettings):

    # -- Default LLM Settings --
    default_model: str = "glm-4.5-air"
    default_temperature: float = 0.0
    default_max_tokens: Optional[int] = 1024
    
    default_timeout: int = 120
    default_max_retries: int = 2
    
    default_system_prompt: str = "you are a helpful assistant"


@lru_cache
def get_config() -> Config:
    return Config()
