"""
OpenAI model utilities for fetching and creating models.
"""

import logging
import os
from typing import Any, no_type_check

import requests
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

logger = logging.getLogger("metalgate_code")

# API configuration
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODELS_ENDPOINT = f"{OPENAI_BASE_URL}/models"

# Default embedding model for Mem0
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def get_mem0_config() -> dict[str, Any]:
    """
    Get Mem0 configuration for OpenAI provider.

    Returns:
        Dictionary with llm and embedder configuration for Mem0.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    embedding_model = os.environ.get("EMBEDDINGS", DEFAULT_EMBEDDING_MODEL)

    config: dict[str, Any] = {
        "llm": {
            "provider": "openai",
            "config": {"api_key": api_key, "model": "gpt-4.1-nano"},
        },
        "embedder": {
            "provider": "openai",
            "config": {"api_key": api_key, "model": embedding_model},
        },
    }
    return config


def fetch_models() -> list[dict[str, str]]:
    """
    Fetch available models from the OpenAI API.

    Returns:
        List of model dictionaries with 'value' and 'name' keys.
        Returns empty list if the fetch fails.
    """
    try:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("No API key found for OpenAI API")
            return []

        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            OPENAI_MODELS_ENDPOINT,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        # Filter for chat models: gpt-* and o* series
        chat_model_ids = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            if model_id.startswith(("gpt-", "o1", "o3-")):
                chat_model_ids.append(model_id)

        return [
            {"value": f"openai:{model_id}", "name": model_id}
            for model_id in sorted(chat_model_ids)
        ]
    except requests.exceptions.RequestException as e:
        logger.warning("Failed to fetch OpenAI models: %s", e)
        return []
    except Exception as e:
        logger.warning("Unexpected error fetching OpenAI models: %s", e)
        return []


@no_type_check
def create_chat_model(
    model_id: str = "openai:gpt-4o",
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """
    Create a LangChain ChatOpenAI instance.

    Args:
        model_id: Model identifier with 'openai:' prefix. Defaults to 'openai:gpt-4o'.
        temperature: Sampling temperature. Defaults to 0.7.
        max_tokens: Maximum tokens to generate. Defaults to None.

    Returns:
        Configured ChatOpenAI instance.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")

    # Strip 'openai:' prefix if present
    model_name = model_id.split(":", 1)[1] if ":" in model_id else model_id

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=SecretStr(api_key) if api_key else None,
    )
