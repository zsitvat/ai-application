from unittest.mock import MagicMock, patch

from src.services.graph.tools.get_position_tool import (
    _filter_document_fields,
    get_position_tool,
    make_position_input_model,
)


def test_filter_document_fields():
    class Doc:
        def __init__(self):
            self._index_name = "idx"
            self.application_id = "app"
            self.starts_at = "now"
            self.id = "id"
            self.payload = "payload"
            self.embedding = "emb"
            self.keep = "keep"

    doc = Doc()
    filtered = _filter_document_fields(doc)
    assert "keep" in filtered
    assert "id" not in filtered


def test_make_position_input_model():
    model = make_position_input_model(["index_name", "city", "field"])
    assert hasattr(model, "__fields__")
    assert "city" in model.__fields__
    assert "field" in model.__fields__


def test_position_search_tool_success():
    position_search_tool = get_position_tool(["city", "field"])
    position_search_func = position_search_tool._run
    mock_redis = MagicMock()
    mock_results = MagicMock()
    mock_results.total = 1
    mock_results.docs = [MagicMock(text="result1"), MagicMock(text="result2")]
    mock_redis.ft.return_value.search.return_value = mock_results
    with patch(
        "src.services.graph.tools.get_position_tool.Redis", return_value=mock_redis
    ):
        result = position_search_func(city="Budapest", field="IT positions", config={})
        assert "result1" in result or "result2" in result


def test_position_search_tool_fallback():
    position_search_tool = get_position_tool(["labels_job_type", "city"])
    position_search_func = position_search_tool._run
    mock_redis = MagicMock()
    mock_results_empty = MagicMock()
    mock_results_empty.total = 0
    mock_results_empty.docs = []
    mock_results_fallback = MagicMock()
    mock_results_fallback.total = 1
    mock_results_fallback.docs = [MagicMock(text="fallback")]
    mock_redis.ft.return_value.search.side_effect = [
        mock_results_empty,
        mock_results_fallback,
    ]
    with patch(
        "src.services.graph.tools.get_position_tool.Redis", return_value=mock_redis
    ):
        result = position_search_func(
            labels_job_type="Engineer", city="Budapest", config={}
        )
        assert "fallback" in result
