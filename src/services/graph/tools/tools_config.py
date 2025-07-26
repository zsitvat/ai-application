from .get_position_tool import get_position_tool
from .issue_tracker_tool import create_issue_tracker_tool
from .retriever_tool import create_vector_retriever_tool
from .search_tool_factory import (
    create_bing_search_tool,
    create_document_link_analyzer_tool,
    create_google_search_tool,
    create_serpapi_search_tool,
    create_tavily_search_tool,
)


from .retriever_tool import redis_vector_search_tool

google_search_tool_instance = create_google_search_tool()
bing_search_tool_instance = create_bing_search_tool()
serpapi_search_tool_instance = create_serpapi_search_tool()
tavily_search_tool_instance = create_tavily_search_tool()
document_link_analyzer_tool_instance = create_document_link_analyzer_tool()
issue_tracker_tool_instance = create_issue_tracker_tool()

AVAILABLE_TOOLS = {
    "redis_vector_search_tool": redis_vector_search_tool,
    "google_search_tool": google_search_tool_instance,
    "bing_search_tool": bing_search_tool_instance,
    "serpapi_search_tool": serpapi_search_tool_instance,
    "tavily_search_tool": tavily_search_tool_instance,
    "document_link_analyzer_tool": document_link_analyzer_tool_instance,
    "issue_tracker_tool": issue_tracker_tool_instance,
    "get_position_tool": get_position_tool,
}
