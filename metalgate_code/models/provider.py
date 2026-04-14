"""
Provider selection and model factory.
"""

import os
from typing import Any, no_type_check

from langchain_core.language_models import BaseChatModel

from metalgate_code.models.anthropic import create_chat_model as create_anthropic_model
from metalgate_code.models.anthropic import fetch_models as fetch_anthropic_models
from metalgate_code.models.anthropic import get_mem0_config as get_anthropic_mem0_config
from metalgate_code.models.evroc import create_chat_model as create_evroc_model
from metalgate_code.models.evroc import fetch_models as fetch_evroc_models
from metalgate_code.models.evroc import get_mem0_config as get_evroc_mem0_config
from metalgate_code.models.openai import create_chat_model as create_openai_model
from metalgate_code.models.openai import fetch_models as fetch_openai_models
from metalgate_code.models.openai import get_mem0_config as get_openai_mem0_config


def get_mem0_config() -> dict[str, Any]:
    """
    Get Mem0 configuration based on the current provider.

    Returns:
        Dictionary with llm and embedder configuration for Mem0.
    """
    provider = get_provider()

    if provider == "evroc":
        return get_evroc_mem0_config()
    elif provider == "openai":
        return get_openai_mem0_config()
    elif provider == "anthropic":
        return get_anthropic_mem0_config()
    else:
        # Fallback to OpenAI configuration
        return get_openai_mem0_config()


def get_provider() -> str:
    """Get the provider from environment variable, defaulting to 'evroc'."""
    return os.environ.get("PROVIDER", "evroc").lower()


@no_type_check
def create_chat_model(model_id: str | None = None) -> BaseChatModel:
    """
    Create a chat model based on the PROVIDER environment variable.

    Args:
        model_id: Optional model identifier. If not provided, uses provider-specific default.

    Returns:
        Configured chat model instance.

    Raises:
        ValueError: If an unsupported provider is specified.
    """
    provider = get_provider()

    if provider == "evroc":
        default_model = "evroc:moonshotai/Kimi-K2.5"
        return create_evroc_model(model_id or default_model)
    elif provider == "openai":
        default_model = "openai:gpt-4o"
        return create_openai_model(model_id or default_model)
    elif provider == "anthropic":
        default_model = "anthropic:claude-3-5-sonnet-20241022"
        return create_anthropic_model(model_id or default_model)
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. Use 'evroc', 'openai', or 'anthropic'."
        )


def fetch_models() -> list[dict[str, str]]:
    """
    Fetch available models from the configured provider.

    Returns:
        List of model dictionaries with 'value' and 'name' keys.
    """
    provider = get_provider()

    if provider == "evroc":
        return fetch_evroc_models()
    elif provider == "openai":
        return fetch_openai_models()
    elif provider == "anthropic":
        return fetch_anthropic_models()
    else:
        return []
