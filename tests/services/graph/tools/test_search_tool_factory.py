from src.services.graph.tools import search_tool_factory
from src.services.graph.tools.web_search_tool import WebSearchTool
from src.services.graph.tools.document_link_analyzer_tool import (
    DocumentLinkAnalyzerTool,
)


def test_create_google_search_tool():
    tool = search_tool_factory.create_google_search_tool(
        api_key="key", cse_id="id", k=3
    )
    assert isinstance(tool, WebSearchTool)
    assert tool.provider.name == "GOOGLE"
    assert tool.k == 3


def test_create_bing_search_tool():
    tool = search_tool_factory.create_bing_search_tool(
        subscription_key="sub", search_url="url", k=2
    )
    assert isinstance(tool, WebSearchTool)
    assert tool.provider.name == "BING"
    assert tool.k == 2


def test_create_serpapi_search_tool():
    tool = search_tool_factory.create_serpapi_search_tool(
        api_key="serp", engine="google", params={"foo": "bar"}, k=1
    )
    assert isinstance(tool, WebSearchTool)
    assert tool.provider.name == "SERPAPI"
    assert tool.k == 1
    assert tool.serpapi_params["engine"] == "google"


def test_create_tavily_search_tool():
    tool = search_tool_factory.create_tavily_search_tool(
        api_key="tavily", topic="news", search_depth="advanced", k=4
    )
    assert isinstance(tool, WebSearchTool)
    assert tool.provider.name == "TAVILY"
    assert tool.k == 4
    assert tool.tavily_topic == "news"
    assert tool.tavily_search_depth == "advanced"


def test_create_document_link_analyzer_tool():
    tool = search_tool_factory.create_document_link_analyzer_tool(
        max_content_length=1234, timeout=10
    )
    assert isinstance(tool, DocumentLinkAnalyzerTool)
    assert tool.max_content_length == 1234
    assert tool.timeout == 10
