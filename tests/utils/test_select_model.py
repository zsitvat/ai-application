import os
from unittest.mock import MagicMock, patch

import pytest

from src.utils.select_model import get_chat_model, get_embedding_model, get_model


def test_get_embedding_model_openai():
    with patch("src.utils.select_model.OpenAIEmbeddings", MagicMock()) as mock_openai:
        model = get_embedding_model(provider="openai", model="ada")
        mock_openai.assert_called_once_with(model="ada")
        assert model is not None


def test_get_embedding_model_azure():
    with patch(
        "src.utils.select_model.AzureOpenAIEmbeddings", MagicMock()
    ) as mock_azure:
        os.environ["AZURE_BASE_URL"] = "http://test"
        os.environ["AZURE_API_VERSION"] = "2023-09-01-preview"
        model = get_embedding_model(provider="azure", deployment="dep", model="ada")
        mock_azure.assert_called_once()
        assert model is not None


def test_get_embedding_model_wrong_provider():
    with patch("src.utils.select_model.logger") as mock_logger:
        with pytest.raises(KeyError):
            get_embedding_model(provider="wrong")
        mock_logger.error.assert_called()


def test_get_chat_model_completions_openai():
    with patch("src.utils.select_model.OpenAI", MagicMock()) as mock_openai:
        model = get_chat_model(
            provider="openai", model="gpt-3.5-turbo", type="completions"
        )
        mock_openai.assert_called_once_with(model="gpt-3.5-turbo", temperature=0)
        assert model is not None


def test_get_chat_model_completions_azure():
    with patch("src.utils.select_model.AzureOpenAI", MagicMock()) as mock_azure:
        os.environ["AZURE_BASE_URL"] = "http://test"
        model = get_chat_model(provider="azure", deployment="dep", type="completions")
        mock_azure.assert_called_once()
        assert model is not None


def test_get_chat_model_chat_openai():
    with patch("src.utils.select_model.ChatOpenAI", MagicMock()) as mock_chat:
        model = get_chat_model(provider="openai", model="gpt-3.5-turbo", type="chat")
        mock_chat.assert_called_once_with(model="gpt-3.5-turbo", temperature=0)
        assert model is not None


def test_get_chat_model_chat_azure():
    with patch(
        "src.utils.select_model.AzureChatOpenAI", MagicMock()
    ) as mock_azure_chat:
        os.environ["AZURE_BASE_URL"] = "http://test"
        model = get_chat_model(provider="azure", deployment="dep", type="chat")
        mock_azure_chat.assert_called_once()
        assert model is not None


def test_get_chat_model_chat_anthropic():
    with patch("src.utils.select_model.ChatAnthropic", MagicMock()) as mock_anthropic:
        model = get_chat_model(provider="anthropic", model="claude-2", type="chat")
        mock_anthropic.assert_called_once_with(
            name="claude-2", temperature=0, timeout=60, stop=None
        )
        assert model is not None


def test_get_chat_model_wrong_provider():
    with patch("src.utils.select_model.logger") as mock_logger:
        with pytest.raises(KeyError):
            get_chat_model(provider="wrong", type="chat")
        mock_logger.error.assert_called()


def test_get_chat_model_wrong_type():
    with patch("src.utils.select_model.logger") as mock_logger:
        with pytest.raises(KeyError):
            get_chat_model(provider="openai", type="wrongtype")
        mock_logger.error.assert_called()


def test_get_chat_model_none_model():
    with pytest.raises(ValueError):
        get_chat_model(provider="openai", model=None, type="chat")


def test_get_model_embedding():
    with patch("src.utils.select_model.OpenAIEmbeddings", MagicMock()) as mock_openai:
        model = get_model(provider="openai", type="embedding", model="test-embedding")
        mock_openai.assert_called_once_with(model="test-embedding")
        assert model is not None


def test_get_model_chat():
    with patch("src.utils.select_model.ChatOpenAI", MagicMock()) as mock_chat:
        model = get_model(provider="openai", type="chat", model="gpt-3.5-turbo")
        mock_chat.assert_called_once_with(model="gpt-3.5-turbo", temperature=0)
        assert model is not None


def test_get_model_invalid_type():
    with pytest.raises(KeyError):
        get_model(provider="openai", type="invalid", model="gpt-3.5-turbo")


def test_get_model_none_model():
    with pytest.raises(ValueError):
        get_model(provider="openai", type="chat", model=None)
