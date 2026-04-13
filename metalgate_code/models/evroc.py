"""
Evroc model utilities for fetching and creating models.
"""

import logging
import os
from typing import no_type_check

import requests
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

logger = logging.getLogger("metalgate_code")

# API configuration
EVROC_BASE_URL = "https://models.think.cloud.evroc.com/v1"
EVROC_MODELS_ENDPOINT = f"{EVROC_BASE_URL}/models"


def fetch_models() -> list[dict[str, str]]:
    """
    Fetch available models from the Evroc API.

    Returns:
        List of model dictionaries with 'value' and 'name' keys.
        Returns empty list if the fetch fails.
    """
    try:
        api_key = os.environ.get("OPENAI_API_KEY", "")
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
    api_key = os.environ.get("OPENAI_API_KEY", "")

    # Strip 'evroc:' prefix if present
    model_name = model_id.split(":", 1)[1] if ":" in model_id else model_id

    return ChatOpenAI(
        model=model_name,
        base_url=EVROC_BASE_URL,
        api_key=SecretStr(api_key) if api_key else None,
    )
