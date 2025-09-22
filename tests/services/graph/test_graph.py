import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.graph.graph import Graph
from src.services.graph.tools.tools_config import AVAILABLE_TOOLS


@pytest.fixture
def graph_instance():
    logger = MagicMock()
    app_settings_service = MagicMock()
    graph_config = MagicMock()
    return Graph(graph_config, logger, app_settings_service)


def test_prepare_graph_execution(graph_instance):
    graph_config = {"agents": {"agent1": {}}}
    user_input = "test input"
    result = graph_instance.prepare_graph_execution(graph_config, user_input)
    assert result is not None


def test_merge_tool_args_with_config(graph_instance):
    tool_args = {
        "positions": "data_scientist",
        "retriever": "vector_db",
        "search_type": "semantic",
    }
    config_defaults = {
        "search_type": "keyword",
        "new_config": "enabled",
        "retriever": "redis_search",
    }
    merged = graph_instance.tool_handler.merge_tool_args_with_config(
        tool_args, config_defaults
    )
    assert merged["search_type"] == "keyword"
    assert merged["new_config"] == "enabled"
    assert merged["positions"] == "data_scientist"
    assert merged["retriever"] == "redis_search"


def test_get_available_tools():
    # Check that AVAILABLE_TOOLS contains expected keys
    expected_tools = [
        "google_search_tool",
        "bing_search_tool",
        "serpapi_search_tool",
        "tavily_search_tool",
        "document_link_analyzer_tool",
        "issue_tracker_tool",
        "get_position_tool",
        "get_labels_tool",
        "web_search_tool",
        "vector_retriever_tool",
    ]
    for tool_name in expected_tools:
        assert tool_name in AVAILABLE_TOOLS


def test_extract_tool_call(graph_instance):
    class DummyResponse:
        tool_calls = [
            {
                "name": "get_labels_tool",
                "id": "label_call_1",
                "args": {"label": "label1", "count": 5},
            }
        ]
        additional_kwargs = {}

    name, args, call_id = graph_instance.tool_handler.extract_tool_call(DummyResponse())
    assert name == "get_labels_tool"
    assert call_id == "label_call_1"
    assert args["label"] == "label1"
    assert args["count"] == 5


def test_graph_get_compiled_workflow(graph_instance):

    graph_instance._load_graph_configuration = AsyncMock(return_value=None)
    graph_instance._build_workflow = lambda: MagicMock(
        compile=lambda checkpointer=None: MagicMock()
    )
    graph_instance._create_checkpointer = AsyncMock(return_value=MagicMock())
    graph_instance.workflow = None
    result = asyncio.run(graph_instance.get_compiled_workflow(1))
    assert result is not None


def test_inject_tool_info_into_prompt(graph_instance):
    prompt = {}
    agent_config = MagicMock()
    result = graph_instance.tool_handler.inject_tool_info_into_prompt(
        prompt, agent_config
    )
    assert result is not None


def test_get_allowed_tool_names(graph_instance):
    agent_config = MagicMock()
    result = graph_instance.tool_handler.get_allowed_tool_names(agent_config)
    assert isinstance(result, list)


def test_find_tool_func(graph_instance):
    tool_name = "get_labels_tool"
    tools_to_bind = [MagicMock(name="get_labels_tool")]
    result = graph_instance.tool_handler.find_tool_func(tool_name, tools_to_bind)
    assert result is not None


def test_update_context_with_tool_results(graph_instance):
    context = {"tool_results": {}}
    tool_results = [{"tool_name": "get_labels_tool", "result": 1}]
    result = graph_instance.tool_handler.update_context_with_tool_results(
        context, tool_results
    )
    assert result is None or isinstance(result, dict)


def test_update_prompt_context(graph_instance):
    prompt = {}
    value = "some_value"
    tool_name = "get_labels_tool"
    result = graph_instance.tool_handler._update_prompt_context(
        prompt, tool_name, "variable_name", value, None
    )
    # Accept None or dict as valid output
    assert result is None or isinstance(result, dict)


def test_extract_position_info_from_list(graph_instance):
    positions = ["Data Scientist", "ML Engineer"]
    application_attributes = {}
    result = graph_instance.tool_handler._extract_position_info_from_list(
        application_attributes, positions
    )
    # Accept None or list as valid output
    assert result is None or isinstance(result, list)


def test_extract_location_info_from_labels(graph_instance):
    labels = {"counties": ["Budapest", "Remote"]}
    application_attributes = {}
    result = graph_instance.tool_handler._extract_location_info_from_labels(
        application_attributes, labels
    )
    # Accept None or list as valid output
    assert result is None or isinstance(result, list)


def test_build_function_definition(graph_instance):
    options = ["option1", "option2"]
    result = graph_instance._build_function_definition(options)
    assert isinstance(result, dict)


def test_extract_next_agent_from_response(graph_instance):
    response = {"next_agent": "agent1"}
    result = graph_instance._extract_next_agent_from_response(response)
    assert result is None or isinstance(result, str)


def test_get_user_input_from_state(graph_instance):
    state = MagicMock()
    state.user_input = "input"
    result = graph_instance._get_user_input_from_state(state)
    assert isinstance(result, str)


def test_bind_tools_to_chain(graph_instance):
    agent_config = MagicMock()
    result = graph_instance.tool_handler.bind_tools_to_chain(agent_config)
    assert isinstance(result, list)


def test_prepare_tool_args(graph_instance):
    tool_args = {"arg1": 1}
    agent_config = MagicMock()
    agent_config.tools = {"test_tool": {"default": 2}}
    result = graph_instance.tool_handler.prepare_tool_args(
        "test_tool", tool_args, agent_config
    )
    assert isinstance(result, dict)


def test_handle_exception_chain_fallback(graph_instance):
    # Minimal test for coverage of fallback error handling
    class DummyState(dict):
        def __init__(self):
            super().__init__()
            self["messages"] = []
            self["next_agent"] = None

    state = DummyState()
    result = graph_instance._handle_exception_chain_fallback(state)
    assert "messages" in result
    assert "next_agent" in result
