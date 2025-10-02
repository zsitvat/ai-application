import os

from langchain_anthropic import ChatAnthropic
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAI,
    OpenAIEmbeddings,
)

from src.config.constants import (
    DEFAULT_API_VERSION,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TIMEOUT,
)
from src.services.logger.logger_service import LoggerService

logger = LoggerService().setup_logger()


def get_model(
    provider: str = "openai",
    deployment: str | None = None,
    model: str | None = None,
    type: str = "chat",
    temperature: float = 0,
) -> object:
    """
    Get the appropriate model instance based on the provided parameters.
    Args:
        provider (str): The model provider (e.g., 'openai', 'azure', 'anthropic')
        deployment (str | None): The deployment name for Azure models
        model (str | None): The specific model name or ID
        type (str): The type of model ('chat', 'completions', 'embeddings')
        temperature (float): The temperature setting for the model
    Returns:
        object: An instance of the selected model class
    """
    if not model:
        model = DEFAULT_CHAT_MODEL if type != "embedding" else DEFAULT_EMBEDDING_MODEL

    if type == "embedding":
        if provider == "openai":
            return OpenAIEmbeddings(model=model)
        elif provider == "azure":
            return AzureOpenAIEmbeddings(
                azure_endpoint=os.environ.get("AZURE_BASE_URL"),
                azure_deployment=deployment,
                api_version=os.environ.get("AZURE_API_VERSION", DEFAULT_API_VERSION),
            )
        else:
            logger.error(f"[SelectModel] Unsupported embedding provider: {provider}")
            raise KeyError(f"Unsupported provider for embedding model: {provider}")

    elif type == "completions":
        if provider == "openai":
            return OpenAI(model=model, temperature=temperature)
        elif provider == "azure":
            return AzureOpenAI(
                azure_endpoint=os.environ.get("AZURE_BASE_URL"),
                azure_deployment=deployment,
                temperature=temperature,
            )
        else:
            logger.error(f"[SelectModel] Unsupported completions provider: {provider}")
            raise KeyError(f"Unsupported provider for completion model: {provider}")

    elif type == "chat":
        if provider == "openai":
            return ChatOpenAI(model=model, temperature=temperature)
        elif provider == "azure":
            return AzureChatOpenAI(
                azure_endpoint=os.environ.get("AZURE_BASE_URL"),
                azure_deployment=deployment,
                temperature=temperature,
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                name=model, temperature=temperature, timeout=DEFAULT_TIMEOUT, stop=None
            )
        else:
            logger.error(f"[SelectModel] Unsupported chat provider: {provider}")
            raise KeyError(f"Unsupported provider for chat model: {provider}")

    else:
        logger.error(f"[SelectModel] Unsupported model type: {type}")
        raise KeyError(f"Unsupported model type: {type}")
