import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.data_api.chat_history import DataChatHistoryService


class DummyRedis:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value


@pytest.fixture
def chat_service():
    service = DataChatHistoryService()
    return service


def test_chat_history_service_init():
    service = DataChatHistoryService()
    assert service is not None


@pytest.mark.asyncio
async def test_get_history_messages_as_string(chat_service):
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
        result = await chat_service.get_history_messages_as_string(uuid="test")
        assert "HUMAN: hi" in result or "AI: hello" in result


@pytest.mark.asyncio
async def test_get_human_messages_as_string(chat_service):
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
        result = await chat_service.get_human_messages_as_string(uuid="test")
        assert "hi" in result


@pytest.mark.asyncio
async def test_get_ai_messages_as_string(chat_service):
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
        result = await chat_service.get_ai_messages_as_string(uuid="test")
        assert "hello" in result


@pytest.mark.asyncio
async def test_get_history_messages_as_dict(chat_service):
    with patch.object(
        chat_service,
        "get_messages_from_chat_history",
        AsyncMock(return_value=[{"type": "ai", "messageAsText": "hello"}]),
    ):
        result = await chat_service.get_history_messages_as_dict(uuid="test")
        assert isinstance(result, list)


@pytest.mark.asyncio
async def test_get_conversations_success(chat_service):
    import os

    os.environ["CHAT_HISTORY_REST_API_ROUTE_PATH"] = (
        "/api/chat/{uuid}?limit={limit}&page={page}&size={size}"
    )
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"content": []}

    class MockAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            # intentionally empty for mock
            pass

        async def get(self, *args, **kwargs):
            await asyncio.sleep(0)  # simulate async
            return mock_response

    with patch("httpx.AsyncClient", return_value=MockAsyncClient()):
        result = await chat_service.get_conversations(uuid="test")
        assert result == mock_response


@pytest.mark.asyncio
async def test_get_conversations_error(chat_service):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.request = MagicMock()

    class MockAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            # intentionally empty for mock
            pass

        async def get(self, *args, **kwargs):
            await asyncio.sleep(0)  # simulate async
            return mock_response

    with patch("httpx.AsyncClient", return_value=MockAsyncClient()):
        with pytest.raises(Exception):
            await chat_service.get_conversations(uuid="test")
