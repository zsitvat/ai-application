from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.graph.tools.get_labels_tool import (
    _build_county_city_field_string,
    get_labels_tool,
)


def test_build_county_city_field_string():
    labels = [
        {"county": "A", "city": "X", "field": "f1"},
        {"county": "A", "city": "X", "field": "f2"},
        {"county": "A", "city": "Y", "field": "f1"},
        {"county": "B", "city": "Z", "field": "f3"},
    ]
    result = _build_county_city_field_string(labels)
    assert "A: X (f1, f2), Y (f1)" in result
    assert "B: Z (f3)" in result


@pytest.mark.asyncio
async def test_get_labels_tool_success():
    mock_redis = AsyncMock()
    mock_results = MagicMock()
    mock_results.docs = [
        MagicMock(labels_county="A", labels_city="X", labels_field="f1"),
        MagicMock(labels_county="A", labels_city="X", labels_field="f2"),
        MagicMock(labels_county="B", labels_city="Y", labels_field="f3"),
    ]
    mock_redis.ft = MagicMock(return_value=mock_redis)
    mock_redis.search = AsyncMock(return_value=mock_results)
    with patch(
        "src.services.graph.tools.get_labels_tool.Redis", return_value=mock_redis
    ):
        result = await get_labels_tool.arun("positions")
        assert "A: X (f1, f2)" in result or "B: Y (f3)" in result
        mock_redis.close.assert_awaited()
