import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.data_api.chat_history import DataChatHistoryService


@pytest.fixture
def chat_service():
    service = DataChatHistoryService()
    return service


def test_get_history_messages_as_string(chat_service):
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
        result = asyncio.run(chat_service.get_history_messages_as_string(uuid="uuid"))
        assert result is not None


def test_get_human_messages_as_string(chat_service):
    with patch.object(
        chat_service,
        "get_messages_from_chat_history",
        AsyncMock(return_value=[{"type": "human", "messageAsText": "hi"}]),
    ):
        result = asyncio.run(chat_service.get_human_messages_as_string(uuid="uuid"))
        assert result is not None


def test_get_ai_messages_as_string(chat_service):
    with patch.object(
        chat_service,
        "get_messages_from_chat_history",
        AsyncMock(return_value=[{"type": "ai", "messageAsText": "hello"}]),
    ):
        result = asyncio.run(chat_service.get_ai_messages_as_string(uuid="uuid"))
        assert result is not None


def test_get_history_messages_as_dict(chat_service):
    with patch.object(
        chat_service,
        "get_messages_from_chat_history",
        AsyncMock(return_value=[{"type": "ai", "messageAsText": "hello"}]),
    ):
        result = asyncio.run(chat_service.get_history_messages_as_dict("uuid"))
        assert isinstance(result, list) or result is None


def test_save_inputs_to_db(chat_service):
    with patch(
        "src.services.data_api.chat_history.PostgresChatMessageHistory"
    ) as mock_pg:
        mock_pg.return_value.add_user_message = MagicMock()
        mock_pg.return_value.add_ai_message = MagicMock()
        chat_service.save_inputs_to_db("q", "a", "session")
        mock_pg.return_value.add_user_message.assert_called_with("q")
        mock_pg.return_value.add_ai_message.assert_called_with("a")


def test_get_conversations(chat_service):
    # Set required env vars for the test
    import os

    os.environ["CHAT_HISTORY_REST_API_ROUTE_PATH"] = (
        "/api/chat/{uuid}?limit={limit}&page={page}&size={size}"
    )
    os.environ["DATA_API_BASE_URL"] = "http://localhost"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"content": []}

    class MockAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def get(self, *args, **kwargs):
            await asyncio.sleep(0)
            return mock_response

    with patch("httpx.AsyncClient", return_value=MockAsyncClient()):
        result = asyncio.run(chat_service.get_conversations("uuid"))
        assert result == mock_response


# New unit tests for edge cases and error handling
def test_get_history_messages_as_string_empty(chat_service):
    with patch.object(
        chat_service,
        "get_messages_from_chat_history",
        AsyncMock(return_value=[]),
    ):
        result = asyncio.run(chat_service.get_history_messages_as_string(uuid="uuid"))
        assert result == ""


def test_get_human_messages_as_string_no_human(chat_service):
    with patch.object(
        chat_service,
        "get_messages_from_chat_history",
        AsyncMock(return_value=[{"type": "ai", "messageAsText": "hello"}]),
    ):
        result = asyncio.run(chat_service.get_human_messages_as_string(uuid="uuid"))
        assert result == ""


def test_get_ai_messages_as_string_no_ai(chat_service):
    with patch.object(
        chat_service,
        "get_messages_from_chat_history",
        AsyncMock(return_value=[{"type": "human", "messageAsText": "hi"}]),
    ):
        result = asyncio.run(chat_service.get_ai_messages_as_string(uuid="uuid"))
        assert result == ""
