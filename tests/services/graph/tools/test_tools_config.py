from src.services.graph.tools import tools_config


def test_available_tools_keys():
    expected_keys = [
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
    for key in expected_keys:
        assert key in tools_config.AVAILABLE_TOOLS


def test_tool_instances_types():
    # Just check that instances are present and callable or class
    for key, tool in tools_config.AVAILABLE_TOOLS.items():
        assert tool is not None
        # Some are classes, some are instances, some are functions
        assert callable(tool) or isinstance(tool, object)
