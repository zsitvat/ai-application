import pytest

from src.schemas.tool_schema import (
    DocumentLinkAnalyzerInput,
    SearchKwargs,
    SearchProvider,
    Tool,
    WebSearchTool,
    WebSearchToolInput,
)


def test_search_provider_enum():
    """Test SearchProvider enum values."""
    assert SearchProvider.GOOGLE == "google"
    assert SearchProvider.BING == "bing"
    assert SearchProvider.SERPAPI == "serpapi"
    assert SearchProvider.TAVILY == "tavily"


def test_search_kwargs():
    """Test SearchKwargs instantiation and field values."""
    obj = SearchKwargs(k=5, threshold=0.5, search_type="similarity")
    assert obj.k == 5
    assert obj.threshold == pytest.approx(0.5)
    assert obj.search_type == "similarity"


def test_tool():
    """Test Tool instantiation and field values."""
    obj = Tool(name="search")
    assert obj.name == "search"


def test_web_search_tool_input():
    """Test WebSearchToolInput instantiation and field values."""
    obj = WebSearchToolInput(query="Find AI jobs")
    assert obj.query == "Find AI jobs"


def test_document_link_analyzer_input():
    """Test DocumentLinkAnalyzerInput instantiation and field values."""
    obj = DocumentLinkAnalyzerInput(
        text="Check these links", max_links=5, include_summary=False
    )
    assert obj.text == "Check these links"
    assert obj.max_links == 5
    assert obj.include_summary is False


def test_web_search_tool():
    """Test WebSearchTool instantiation and field values."""
    obj = WebSearchTool(name="web_search", provider=SearchProvider.BING)
    assert obj.name == "web_search"
    assert obj.provider == SearchProvider.BING
