"""
Evroc model utilities for fetching and creating models.
"""

import logging
import os
from typing import Any, no_type_check

import requests
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

logger = logging.getLogger("metalgate_code")

# API configuration
EVROC_BASE_URL = "https://models.think.cloud.evroc.com/v1"
EVROC_MODELS_ENDPOINT = f"{EVROC_BASE_URL}/models"


def get_mem0_config() -> dict[str, Any]:
    """
    Get Mem0 configuration for Evroc provider.

    Evroc is OpenAI-compatible, so we use OpenAI-compatible config
    for both LLM and embedder.

    Returns:
        Dictionary with llm and embedder configuration for Mem0.
    """
    api_key = os.environ.get("MODEL_API_KEY", "")
    embedder_api_key = os.environ.get("EMBEDDER_API_KEY", "")
    mem_model = os.environ.get("MEM_MODEL", "moonshotai/Kimi-K2.5")
    temperature = os.environ.get("TEMPERATURE", 0.6)
    embedder_model = os.environ.get("MEM_EMBEDDER_MODEL", "Qwen/Qwen3-Embedding-8B")

    config: dict[str, Any] = {
        "llm": {
            "provider": "openai",
            "config": {
                "api_key": api_key,
                "model": mem_model,
                "openai_base_url": EVROC_BASE_URL,
                "temperature": temperature,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "api_key": embedder_api_key,
                "model": embedder_model,
                "openai_base_url": EVROC_BASE_URL,
            },
        },
    }
    return config


def fetch_models() -> list[dict[str, str]]:
    """
    Fetch available models from the Evroc API.

    Returns:
        List of model dictionaries with 'value' and 'name' keys.
        Returns empty list if the fetch fails.
    """
    try:
        api_key = os.environ.get("MODEL_API_KEY", "")
        if not api_key:
            logger.warning("No API key found for Evroc API")
            return []

        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            EVROC_MODELS_ENDPOINT,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        return [
            {"value": f"evroc:{model['id']}", "name": model.get("name", model["id"])}
            for model in data.get("data", [])
        ]
    except requests.exceptions.RequestException as e:
        logger.warning("Failed to fetch Evroc models: %s", e)
        return []
    except Exception as e:
        logger.warning("Unexpected error fetching Evroc models: %s", e)
        return []


@no_type_check
def create_chat_model(model_id: str = "evroc:moonshotai/Kimi-K2.5") -> ChatOpenAI:
    """
    Create a LangChain ChatOpenAI instance for Evroc models.

    Args:
        model_id: Model identifier with 'evroc:' prefix. Defaults to 'evroc:moonshotai/Kimi-K2.5'.

    Returns:
        Configured ChatOpenAI instance for the Evroc API.
    """
    api_key = os.environ.get("MODEL_API_KEY", "")
    temperature = os.environ.get("TEMPERATURE", 0.6)

    # Strip 'evroc:' prefix if present
    model_name = model_id.split(":", 1)[1] if ":" in model_id else model_id

    return ChatOpenAI(
        model=model_name,
        base_url=EVROC_BASE_URL,
        api_key=SecretStr(api_key) if api_key else None,
        temperature=temperature,
    )
