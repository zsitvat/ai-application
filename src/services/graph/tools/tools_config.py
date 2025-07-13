from .document_link_analyzer_tool import DocumentLinkAnalyzerTool
from .retriever_tool import create_vector_retriever_tool
from .search_tool_factory import (
    create_bing_search_tool,
    create_document_link_analyzer_tool,
    create_google_search_tool,
    create_serpapi_search_tool,
    create_tavily_search_tool,
)
from .web_search_tool import WebSearchTool

AVAILABLE_TOOLS = {
    "create_vector_retriever_tool": create_vector_retriever_tool,
    "create_google_search_tool": create_google_search_tool,
    "create_bing_search_tool": create_bing_search_tool,
    "create_serpapi_search_tool": create_serpapi_search_tool,
    "create_tavily_search_tool": create_tavily_search_tool,
    "create_document_link_analyzer_tool": create_document_link_analyzer_tool,
}
