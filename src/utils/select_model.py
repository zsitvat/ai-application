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
    ANTHROPIC_PROVIDER,
    AZURE_PROVIDER,
    DEFAULT_API_VERSION,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TIMEOUT,
    OPENAI_PROVIDER,
)
from src.services.logger.logger_service import LoggerService

# Type alias for all supported model types
ModelType = (
    OpenAI
    | AzureOpenAI
    | ChatOpenAI
    | AzureChatOpenAI
    | ChatAnthropic
    | OpenAIEmbeddings
    | AzureOpenAIEmbeddings
)

logger = LoggerService().setup_logger()


def _get_azure_config() -> dict[str, str]:
    """Get Azure configuration from environment variables."""
    return {
        "azure_endpoint": os.environ.get("AZURE_BASE_URL"),
        "api_version": os.environ.get("AZURE_API_VERSION", DEFAULT_API_VERSION),
    }


def _create_openai_completion_model(model: str, temperature: float) -> OpenAI:
    """Create OpenAI completion model."""
    if model is None:
        raise ValueError("Model cannot be None")
    return OpenAI(model=model, temperature=temperature)


def _create_azure_completion_model(deployment: str, temperature: float) -> AzureOpenAI:
    """Create Azure OpenAI completion model."""
    azure_config = _get_azure_config()
    return AzureOpenAI(
        azure_endpoint=azure_config["azure_endpoint"],
        azure_deployment=deployment,
        temperature=temperature,
    )


def _create_openai_chat_model(model: str, temperature: float) -> ChatOpenAI:
    """Create OpenAI chat model."""
    if model is None:
        raise ValueError("Model cannot be None")
    return ChatOpenAI(model=model, temperature=temperature)


def _create_azure_chat_model(deployment: str, temperature: float) -> AzureChatOpenAI:
    """Create Azure OpenAI chat model."""
    azure_config = _get_azure_config()
    return AzureChatOpenAI(
        azure_endpoint=azure_config["azure_endpoint"],
        azure_deployment=deployment,
        temperature=temperature,
    )


def _create_anthropic_chat_model(model: str, temperature: float) -> ChatAnthropic:
    """Create Anthropic chat model."""
    if model is None:
        raise ValueError("Model cannot be None")
    return ChatAnthropic(
        name=model, temperature=temperature, timeout=DEFAULT_TIMEOUT, stop=None
    )


def get_embedding_model(
    provider: str = OPENAI_PROVIDER,
    deployment: str | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> AzureOpenAIEmbeddings | OpenAIEmbeddings:
    """Get an embedding model based on the provider

    Args:
        provider: The provider of the model (openai or azure)
        deployment: The deployment of the model (for Azure)
        model: The model name

    Returns:
        AzureOpenAIEmbeddings | OpenAIEmbeddings: An embedding model instance

    Raises:
        KeyError: If unsupported provider is specified
    """
    if provider == OPENAI_PROVIDER:
        return OpenAIEmbeddings(model=model)
    elif provider == AZURE_PROVIDER:
        azure_config = _get_azure_config()
        return AzureOpenAIEmbeddings(
            azure_endpoint=azure_config["azure_endpoint"],
            azure_deployment=deployment,
            api_version=azure_config["api_version"],
        )
    else:
        logger.error(f"[SelectModel] Wrong embedding model provider: {provider}")
        raise KeyError(f"Unsupported provider for embedding model: {provider}")


def get_chat_model(
    provider: str,
    deployment: str | None = None,
    model: str | None = DEFAULT_CHAT_MODEL,
    type: str = "chat",
    temperature: float = 0,
) -> ModelType:
    """Get the model based on the provider and the type of the model

    Args:
        provider (str): The provider of the model
        deployment (str | None, optional): The deployment of the model. Defaults to None.
        model (str, optional): The model name. Defaults to "gpt-4o-mini".
        type (str, optional): The type of the model. Defaults to "chat".
        temperature (float, optional): The temperature of the model. Defaults to 0.

    Returns:
        ModelType: A model class instance

    Raises:
        KeyError: If unsupported provider or type is specified
        ValueError: If model is None when required
    """
    if type == "completions":
        return _get_completion_model(provider, deployment, model, temperature)
    elif type == "chat":
        return _get_chat_model(provider, deployment, model, temperature)
    else:
        logger.error(f"Wrong model type: {type}")
        raise KeyError(f"Unsupported model type: {type}")


def _get_completion_model(
    provider: str, deployment: str | None, model: str | None, temperature: float
) -> OpenAI | AzureOpenAI:
    """Get completion model based on provider."""
    if provider == OPENAI_PROVIDER:
        return _create_openai_completion_model(model, temperature)
    elif provider == AZURE_PROVIDER:
        return _create_azure_completion_model(deployment, temperature)
    else:
        logger.error(f"Wrong model provider for completions: {provider}")
        raise KeyError(f"Unsupported provider for completion model: {provider}")


def _get_chat_model(
    provider: str, deployment: str | None, model: str | None, temperature: float
) -> ChatOpenAI | AzureChatOpenAI | ChatAnthropic:
    """Get chat model based on provider."""
    if provider == OPENAI_PROVIDER:
        return _create_openai_chat_model(model, temperature)
    elif provider == AZURE_PROVIDER:
        return _create_azure_chat_model(deployment, temperature)
    elif provider == ANTHROPIC_PROVIDER:
        return _create_anthropic_chat_model(model, temperature)
    else:
        logger.error(f"Wrong model provider for chat: {provider}")
        raise KeyError(f"Unsupported provider for chat model: {provider}")


def get_model(
    provider: str,
    deployment: str | None = None,
    model: str | None = DEFAULT_CHAT_MODEL,
    type: str = "chat",
    temperature: float = 0,
) -> ModelType:
    """Get the model based on the provider, deployment, model name, type, and temperature.

    Args:
        provider: The provider of the model (openai, azure, anthropic)
        deployment: The deployment of the model (for Azure)
        model: The model name
        type: The type of the model (chat, completions, embedding)
        temperature: The temperature of the model

    Returns:
        ModelType: A model instance

    Raises:
        KeyError: If unsupported provider or type is specified
        ValueError: If model is None when required
    """
    if type == "embedding":
        embedding_model = model if model is not None else DEFAULT_EMBEDDING_MODEL
        return get_embedding_model(provider, deployment, embedding_model)
    else:
        return get_chat_model(provider, deployment, model, type, temperature)
