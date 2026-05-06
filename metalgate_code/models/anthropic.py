"""
Anthropic model utilities for fetching and creating models.
"""

import logging
import os
from typing import Any, no_type_check

import requests
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

logger = logging.getLogger("metalgate_code")

# API configuration
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_MODELS_ENDPOINT = f"{ANTHROPIC_BASE_URL}/models"


def get_mem0_config() -> dict[str, Any]:
    """
    Get Mem0 configuration for Anthropic provider.

    Anthropic doesn't provide embeddings, so we use OpenAI embeddings
    with Anthropic LLM for Mem0.

    Returns:
        Dictionary with llm and embedder configuration for Mem0.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    embedder_api_key = os.environ.get("EMBEDDER_API_KEY", "")
    mem_model = os.environ.get("MEM_MODEL", "claude-3-5-haiku-20241022")
    temperature = os.environ.get("TEMPERATURE", 0.7)
    embedder_model = os.environ.get("MEM_EMBEDDER_MODEL", "voyage-3.5-lite")

    config: dict[str, Any] = {
        "llm": {
            "provider": "anthropic",
            "config": {
                "api_key": api_key,
                "model": mem_model,
                "temperature": temperature,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {"api_key": embedder_api_key, "model": embedder_model},
        },
    }
    return config


def fetch_models() -> list[dict[str, str]]:
    """
    Fetch available models from the Anthropic API.

    Returns:
        List of model dictionaries with 'value' and 'name' keys.
        Returns empty list if the fetch fails.
    """
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.warning("No API key found for Anthropic API")
            return []

        headers = {
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
        }
        response = requests.get(
            ANTHROPIC_MODELS_ENDPOINT,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        # Filter for Claude models
        model_ids = [
            model["id"]
            for model in data.get("data", [])
            if model.get("id", "").startswith("claude-")
        ]

        return [
            {"value": f"anthropic:{model_id}", "name": model_id}
            for model_id in sorted(model_ids)
        ]
    except requests.exceptions.RequestException as e:
        logger.warning("Failed to fetch Anthropic models: %s", e)
        return []
    except Exception as e:
        logger.warning("Unexpected error fetching Anthropic models: %s", e)
        return []


@no_type_check
def create_chat_model(
    model_id: str = "anthropic:claude-3-5-sonnet-20241022",
) -> ChatAnthropic:
    """
    Create a LangChain ChatAnthropic instance.

    Args:
        model_id: Model identifier with 'anthropic:' prefix.
                  Defaults to 'anthropic:claude-3-5-sonnet-20241022'.

    Returns:
        Configured ChatAnthropic instance.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    temperature = os.environ.get("TEMPERATURE", 0.7)
    max_tokens = os.environ.get("MAX_TOKENS", None)

    # Strip 'anthropic:' prefix if present
    model_name = model_id.split(":", 1)[1] if ":" in model_id else model_id

    return ChatAnthropic(
        model=model_name,
        anthropic_api_key=SecretStr(api_key) if api_key else None,
        temperature=temperature,
        max_tokens=max_tokens,
    )
