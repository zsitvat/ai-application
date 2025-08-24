import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from src.services.graph.graph_service import GraphService


@pytest.mark.asyncio
async def test_graph_service_execute_graph():
    mock_graph = MagicMock()
    mock_graph.prepare_graph_execution = AsyncMock(return_value="user_input")
    mock_graph.graph_config = MagicMock(recursion_limit=1)
    mock_graph.workflow.ainvoke = AsyncMock(return_value=MagicMock())
    mock_graph._generate_final_response = MagicMock(return_value="final_response")
    mock_graph._handle_execution_error = AsyncMock(return_value="error_response")
    app_settings_service = MagicMock()
    service = GraphService(app_settings_service, mock_graph)
    result = await service.execute_graph("input", 1, "user_id")
    assert result == "final_response"


@pytest.mark.asyncio
async def test_graph_service_execute_graph_stream():
    mock_graph = MagicMock()
    mock_graph.prepare_graph_execution = AsyncMock(return_value="user_input")
    mock_graph.graph_config = MagicMock(recursion_limit=1)
    mock_graph.workflow.ainvoke = AsyncMock(return_value=MagicMock())
    mock_graph._generate_final_response = MagicMock(return_value="token1 token2")

    async def async_tokenize(text):
        for token in ["token1 ", "token2 "]:
            yield token

    mock_graph._tokenize = async_tokenize
    mock_graph._handle_execution_error = AsyncMock(return_value="error_response")
    app_settings_service = MagicMock()
    service = GraphService(app_settings_service, mock_graph)
    tokens = [t async for t in service.execute_graph_stream("input", 1, "user_id")]
    assert tokens == ["token1 ", "token2 "]


@pytest.mark.asyncio
async def test_graph_service_execute_graph_stream_error():
    mock_graph = MagicMock()
    mock_graph.prepare_graph_execution = AsyncMock(side_effect=Exception("fail"))
    mock_graph._handle_execution_error = AsyncMock(return_value="error_response")

    class AsyncIter:
        def __aiter__(self):
            self._iter = iter(["error "])
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

    mock_graph._tokenize = MagicMock(return_value=AsyncIter())
    mock_graph.graph_config = MagicMock(recursion_limit=1)
    app_settings_service = MagicMock()
    service = GraphService(app_settings_service, mock_graph)
    tokens = [t async for t in service.execute_graph_stream("input", 1, "user_id")]
    assert tokens == ["error "]


@pytest.mark.asyncio
async def test_graph_service_tokenize():
    app_settings_service = MagicMock()
    graph = MagicMock()
    service = GraphService(app_settings_service, graph)
    text = "token1 token2"
    tokens = [t async for t in service._tokenize(text)]
    assert tokens == ["token1 ", "token2 "]


def test_graph_service_init():
    app_settings_service = MagicMock()
    graph = MagicMock()
    service = GraphService(app_settings_service, graph)
    assert service.app_settings_service is app_settings_service
    assert service.graph is graph


@pytest.mark.asyncio
async def test_graph_service_execute_graph_error():
    mock_graph = MagicMock()
    mock_graph.prepare_graph_execution = AsyncMock(side_effect=Exception("fail"))
    mock_graph._handle_execution_error = AsyncMock(return_value="error_response")
    mock_graph.graph_config = MagicMock(recursion_limit=1)
    app_settings_service = MagicMock()
    service = GraphService(app_settings_service, mock_graph)
    result = await service.execute_graph("input", 1, "user_id")
    assert result == "error_response"
