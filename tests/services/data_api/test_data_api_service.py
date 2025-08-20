import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.services.data_api.chat_history import DataChatHistoryService


@pytest.fixture
def chat_service():
    return DataChatHistoryService()


@pytest.mark.asyncio
def test_get_history_messages_as_list(chat_service):
    with patch.object(
        chat_service,
        "get_messages_from_chat_history",
        AsyncMock(
            return_value=[
                {"type": "human", "messageAsText": "hi"},
                {"type": "ai", "messageAsText": "hello"},
            ]
        ),
    ):
        result = asyncio.run(chat_service.get_history_messages_as_list(uuid="uuid"))
        assert len(result) == 2
        assert result[0].content == "hi" or result[1].content == "hello"


@pytest.mark.asyncio
def test_get_ai_messages_as_list(chat_service):
    with patch.object(
        chat_service,
        "get_messages_from_chat_history",
        AsyncMock(
            return_value=[
                {"type": "ai", "messageAsText": "hello"},
                {"type": "human", "messageAsText": "hi"},
            ]
        ),
    ):
        result = asyncio.run(chat_service.get_ai_messages_as_list(uuid="uuid"))
        assert len(result) == 1
        assert result[0].content == "hello"


@pytest.mark.asyncio
def test_get_human_messages_as_list(chat_service):
    with patch.object(
        chat_service,
        "get_messages_from_chat_history",
        AsyncMock(
            return_value=[
                {"type": "human", "messageAsText": "hi"},
                {"type": "ai", "messageAsText": "hello"},
            ]
        ),
    ):
        result = asyncio.run(chat_service.get_human_messages_as_list(uuid="uuid"))
        assert len(result) == 1
        assert result[0].content == "hi"


@pytest.mark.asyncio
def test_get_ai_messages(chat_service):
    with patch.object(
        chat_service,
        "get_messages_from_chat_history",
        AsyncMock(
            return_value=[
                {"type": "ai", "messageAsText": "hello"},
                {"type": "human", "messageAsText": "hi"},
            ]
        ),
    ):
        result = asyncio.run(chat_service.get_ai_messages(uuid="uuid"))
        assert result == ["hello"]


@pytest.mark.asyncio
def test_get_human_messages(chat_service):
    with patch.object(
        chat_service,
        "get_messages_from_chat_history",
        AsyncMock(
            return_value=[
                {"type": "human", "messageAsText": "hi"},
                {"type": "ai", "messageAsText": "hello"},
            ]
        ),
    ):
        result = asyncio.run(chat_service.get_human_messages(uuid="uuid"))
        assert result == ["hi"]


@pytest.mark.asyncio
def test_get_messages_from_chat_history_error(chat_service):
    with patch.object(
        chat_service,
        "get_conversations",
        AsyncMock(return_value=None),
    ):
        with pytest.raises(ValueError):
            asyncio.run(chat_service.get_messages_from_chat_history(uuid="uuid"))
