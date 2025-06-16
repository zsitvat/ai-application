import asyncio
import logging
import os

import boto3
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_community.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_community.llms.openai import OpenAI
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAI,
    OpenAIEmbeddings,
)


async def get_embedding_model(
    provider: str, deployment: str | None = None, model: str = "text-embedding-3-large"
) -> AzureOpenAIEmbeddings | OpenAIEmbeddings:
    """Get an embedding model based on the provider

    Args:
        provider (str): The provider of the model (openai or azure)
        deployment (str | None, optional): The deployment of the model. Defaults to None.
        model (str, optional): The model name. Defaults to "text-embedding-ada-002".

    Returns:
        OpenAIEmbeddings | AzureOpenAIEmbeddings: An embedding model instance
    """
    if provider == "openai":
        return OpenAIEmbeddings(model=model)
    elif provider == "azure":
        return AzureOpenAIEmbeddings(
            azure_endpoint=os.environ.get("AZURE_BASE_URL"),
            azure_deployment=deployment,
        )
    else:
        logging.getLogger("logger").error("Wrong model provider!")
        raise KeyError(f"Unsupported provider for embedding model: {provider}")


async def get_conversation_model(
    provider: str,
    deployment: str | None = None,
    model: str = "gpt-4o-mini",
    type: str = "chat",
    temperature: float = 0,
) -> OpenAI | AzureOpenAI | ChatOpenAI | AzureChatOpenAI | ChatAnthropic | ChatBedrock:
    """Get the model based on the provider and the type of the model

    Args:
        provider (str): The provider of the model
        deployment (str | None, optional): The deployment of the model. Defaults to None.
        model (str, optional): The model name. Defaults to "gpt-4o-mini".
        type (str, optional): The type of the model. Defaults to "completions".
        temperature (float, optional): The temperature of the model. Defaults to 0.

    Returns:
        OpenAI | AzureOpenAI | ChatOpenAI | AzureChatOpenAI | ChatAnthropic | ChatBedrock modell class
    """

    if type == "completions":
        if provider == "openai":
            if model is None:
                raise ValueError("Model cannot be None")
            return OpenAI(model=model, temperature=temperature)
        elif provider == "azure":
            return AzureOpenAI(
                azure_endpoint=os.environ.get("AZURE_BASE_URL"),
                azure_deployment=deployment,
                temperature=temperature,
            )
    if type == "chat":
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
                model_name=model, temperature=temperature, timeout=60, stop=None
            )
        elif provider == "bedrock":
            client = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.environ.get("BEDROCK_AWS_REGION_NAME"),
                aws_access_key_id=os.environ.get("BEDROCK_AWS_ACCESS_KEY"),
                aws_secret_access_key=os.environ.get("BEDROCK_AWS_SECRET_KEY"),
            )

            return ChatBedrock(
                client=client, model=model, model_kwargs=dict(temperature=temperature)
            )
        else:
            logging.getLogger("logger").error("Wrong model provider!")
            raise KeyError(f"Unsupported provider for chat model: {provider}")
    else:
        logging.getLogger("logger").error("Wrong model type!")
        raise KeyError(f"Unsupported model type: {type}")


async def get_model(
    provider: str,
    deployment: str | None = None,
    model: str = "gpt-4o-mini",
    type: str = "chat",
    temperature: float = 0,
) -> (
    OpenAI
    | AzureOpenAI
    | ChatOpenAI
    | AzureChatOpenAI
    | ChatAnthropic
    | ChatBedrock
    | OpenAIEmbeddings
    | AzureOpenAIEmbeddings
):
    """Alias for get_conversation_model for backward compatibility."""
    if type == "embedding":
        return await get_embedding_model(provider, deployment, model)
    else:
        return await get_conversation_model(
            provider, deployment, model, type, temperature
        )
