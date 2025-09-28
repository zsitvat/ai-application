from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.chat_history.redis_chat_history import RedisChatHistoryService


@pytest.mark.asyncio
async def test_get_chat_histories_success():
    svc = RedisChatHistoryService()
    mock_redis = MagicMock()
    mock_redis.scan = AsyncMock(side_effect=[(0, [b"key1"]), (0, [])])
    mock_redis.lrange = AsyncMock(
        return_value=[
            b'{"data": {"additional_kwargs": {"title": "t", "timestamp": "2024-01-01T00:00:00"}}}'
        ]
    )
    mock_redis.close = AsyncMock()
    with patch(
        "src.services.chat_history.redis_chat_history.redis.from_url",
        return_value=mock_redis,
    ):
        result = await svc.get_chat_histories("db", "idx", "user")
        assert isinstance(result, list)
        assert result[0]["title"] == "t"


@pytest.mark.asyncio
async def test_get_chat_histories_empty():
    svc = RedisChatHistoryService()
    mock_redis = MagicMock()
    mock_redis.scan = AsyncMock(side_effect=[(0, [])])
    mock_redis.close = AsyncMock()
    with patch(
        "src.services.chat_history.redis_chat_history.redis.from_url",
        return_value=mock_redis,
    ):
        result = await svc.get_chat_histories("db", "idx", "user")
        assert result == []


@pytest.mark.asyncio
async def test_create_chat_history_success():
    svc = RedisChatHistoryService()
    mock_redis = MagicMock()
    mock_redis.rpush = AsyncMock()
    mock_redis.close = AsyncMock()
    with patch(
        "src.services.chat_history.redis_chat_history.redis.from_url",
        return_value=mock_redis,
    ):
        elements = [{"foo": "bar"}]
        result = await svc.create_chat_history("db", "idx", "user", "sess", elements)
        assert result == elements


@pytest.mark.asyncio
async def test_get_chat_history_success():
    svc = RedisChatHistoryService()
    mock_redis = MagicMock()
    mock_redis.lrange = AsyncMock(return_value=[b'{"foo": "bar"}'])
    mock_redis.close = AsyncMock()
    with patch(
        "src.services.chat_history.redis_chat_history.redis.from_url",
        return_value=mock_redis,
    ):
        result = await svc.get_chat_history("db", "idx", "user", "sess")
        assert result[0]["foo"] == "bar"


@pytest.mark.asyncio
async def test_delete_chat_history_session():
    svc = RedisChatHistoryService()
    mock_redis = MagicMock()
    mock_redis.delete = AsyncMock()
    mock_redis.close = AsyncMock()
    with patch(
        "src.services.chat_history.redis_chat_history.redis.from_url",
        return_value=mock_redis,
    ):
        result = await svc.delete_chat_history("db", "idx", "user", "sess")
        assert result is True


@pytest.mark.asyncio
async def test_delete_chat_history_all():
    svc = RedisChatHistoryService()
    mock_redis = MagicMock()

    class AsyncIter:
        def __init__(self, items):
            self.items = items

        def __aiter__(self):
            self._iter = iter(self.items)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

    mock_redis.scan_iter = lambda *args, **kwargs: AsyncIter([b"key1", b"key2"])
    mock_redis.delete = AsyncMock()
    mock_redis.close = AsyncMock()
    with patch(
        "src.services.chat_history.redis_chat_history.redis.from_url",
        return_value=mock_redis,
    ):
        result = await svc.delete_chat_history("db", "idx", "user", None)
        assert result is True


@pytest.mark.asyncio
async def test_get_chat_histories_error():
    svc = RedisChatHistoryService()
    mock_redis = MagicMock()
    mock_redis.scan = AsyncMock(side_effect=Exception("fail"))
    mock_redis.close = AsyncMock()
    with patch(
        "src.services.chat_history.redis_chat_history.redis.from_url",
        return_value=mock_redis,
    ):
        with pytest.raises(Exception):
            await svc.get_chat_histories("db", "idx", "user")


@pytest.mark.asyncio
async def test_create_chat_history_error():
    svc = RedisChatHistoryService()
    mock_redis = MagicMock()
    mock_redis.rpush = AsyncMock(side_effect=Exception("fail"))
    mock_redis.close = AsyncMock()
    with patch(
        "src.services.chat_history.redis_chat_history.redis.from_url",
        return_value=mock_redis,
    ):
        with pytest.raises(Exception):
            await svc.create_chat_history("db", "idx", "user", "sess", [{"foo": "bar"}])
