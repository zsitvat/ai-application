import os

from schemas.tool_schema import SearchProvider

from .document_link_analyzer_tool import DocumentLinkAnalyzerTool
from .retriever_tool import create_vector_retriever_tool
from .web_search_tool import WebSearchTool


def create_google_search_tool(
    api_key: str | None = None, cse_id: str | None = None, k: int = 5
) -> WebSearchTool:
    """Create a Google Search tool."""
    return WebSearchTool(
        provider=SearchProvider.GOOGLE,
        google_api_key=api_key,
        google_cse_id=cse_id,
        k=k,
    )


def create_bing_search_tool(
    subscription_key: str | None = None, search_url: str | None = None, k: int = 5
) -> WebSearchTool:
    """Create a Bing Search tool."""
    return WebSearchTool(
        provider=SearchProvider.BING,
        bing_subscription_key=subscription_key,
        bing_search_url=search_url
        or os.getenv("BING_SEARCH_URL", "https://api.bing.microsoft.com/v7.0/search"),
        k=k,
    )


def create_serpapi_search_tool(
    api_key: str | None = None,
    engine: str = "google",
    params: dict[str, any] | None = None,
    k: int = 5,
) -> WebSearchTool:
    """Create a SerpAPI Search tool."""
    serpapi_params = params or {}
    if engine:
        serpapi_params["engine"] = engine

    return WebSearchTool(
        provider=SearchProvider.SERPAPI,
        serpapi_api_key=api_key,
        serpapi_params=serpapi_params,
        k=k,
    )


def create_tavily_search_tool(
    api_key: str | None = None,
    topic: str = "general",
    search_depth: str = "basic",
    k: int = 5,
) -> WebSearchTool:
    """Create a Tavily Search tool."""
    return WebSearchTool(
        provider=SearchProvider.TAVILY,
        tavily_api_key=api_key,
        tavily_topic=topic,
        tavily_search_depth=search_depth,
        k=k,
    )


def create_document_link_analyzer_tool(
    max_content_length: int = 5000,
    timeout: int = 30,
) -> DocumentLinkAnalyzerTool:
    """Create a Document Link Analyzer tool."""
    return DocumentLinkAnalyzerTool(
        max_content_length=max_content_length,
        timeout=timeout,
    )
