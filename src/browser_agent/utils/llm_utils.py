"""LLM client utilities for browser automation agent."""

import os
from typing import Optional
from langchain_core.language_models import BaseChatModel

from browser_agent.agent.configuration import (
    LLM_TYPE,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    OPENAI_API_KEY,
    LLM_API_VERSION,
    LLM_MODEL_NAME,
)
from browser_agent.utils.log_utils import get_logger

logger = get_logger(__name__)

# Gemini API Key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def create_llm_client(
    llm_type: Optional[str] = None,
    model_name: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    api_version: Optional[str] = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Create an LLM client based on configuration.

    Args:
        llm_type: Type of LLM ("azure_openai", "openai", or "gemini")
        model_name: Model name/deployment name
        azure_endpoint: Azure OpenAI endpoint
        azure_api_key: Azure OpenAI API key
        openai_api_key: OpenAI API key
        google_api_key: Google Gemini API key
        api_version: API version for Azure
        temperature: Sampling temperature

    Returns:
        Configured LLM client

    Raises:
        ValueError: If required credentials are missing
    """
    llm_type = llm_type or LLM_TYPE
    model_name = model_name or LLM_MODEL_NAME

    if llm_type == "azure_openai":
        from langchain_openai import AzureChatOpenAI

        endpoint = azure_endpoint or AZURE_OPENAI_ENDPOINT
        api_key = azure_api_key or AZURE_OPENAI_API_KEY
        version = api_version or LLM_API_VERSION

        if not endpoint or not api_key:
            raise ValueError(
                "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
            )

        logger.info(f"Creating Azure OpenAI client with model: {model_name}")

        return AzureChatOpenAI(
            azure_deployment=model_name,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=version,
            temperature=temperature,
        )

    elif llm_type == "openai":
        from langchain_openai import ChatOpenAI

        api_key = openai_api_key or OPENAI_API_KEY

        if not api_key:
            raise ValueError("OpenAI requires OPENAI_API_KEY")

        logger.info(f"Creating OpenAI client with model: {model_name}")

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
        )

    elif llm_type == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = google_api_key or GOOGLE_API_KEY

        if not api_key:
            raise ValueError("Gemini requires GOOGLE_API_KEY")

        # Default to gemini-2.0-flash for good compatibility
        if model_name in ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "gemini-pro"]:
            model_name = "gemini-2.0-flash"

        # Ensure model name has correct format
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        logger.info(f"Creating Gemini client with model: {model_name}")

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unknown LLM type: {llm_type}. Use 'azure_openai', 'openai', or 'gemini'")


def create_llm_with_tools(tools: list, **kwargs) -> BaseChatModel:
    """Create an LLM client with tools bound.

    Args:
        tools: List of LangChain tools to bind
        **kwargs: Additional arguments passed to create_llm_client

    Returns:
        LLM client with tools bound
    """
    llm = create_llm_client(**kwargs)
    return llm.bind_tools(tools, tool_choice="required")
