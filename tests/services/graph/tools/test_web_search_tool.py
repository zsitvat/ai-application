from src.schemas.tool_schema import SearchProvider, WebSearchToolInput
from src.services.graph.tools.web_search_tool import WebSearchTool


def test_web_search_tool_init():
    tool = WebSearchTool(
        provider=SearchProvider.GOOGLE, k=2, google_api_key="key", google_cse_id="id"
    )
    assert tool.provider == SearchProvider.GOOGLE
    assert tool.k == 2
    assert tool.google_api_key == "key"
    assert tool.google_cse_id == "id"


def test_web_search_tool_args_schema():
    assert hasattr(WebSearchTool, "args_schema")
    assert issubclass(WebSearchTool.args_schema, WebSearchToolInput)


def test_web_search_tool_str():
    tool = WebSearchTool()
    assert "bing_search_url" in str(tool)
