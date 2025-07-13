from enum import Enum

from pydantic import BaseModel, Field


class SearchProvider(str, Enum):
    GOOGLE = "google"
    BING = "bing"
    SERPAPI = "serpapi"
    TAVILY = "tavily"


class SearchKwargs(BaseModel):
    k: int
    threshold: float = 0.5
    search_type: str = "similarity"


class Tool(BaseModel):
    name: str


class WebSearchToolInput(BaseModel):
    """Input schema for web search tool."""

    query: str = Field(description="The search query")


class DocumentLinkAnalyzerInput(BaseModel):
    """Input schema for document link analyzer tool."""

    text: str = Field(
        description="Text content or prompt that may contain URLs to analyze"
    )
    max_links: int = Field(default=10, description="Maximum number of links to process")
    include_summary: bool = Field(
        default=True, description="Whether to include content summary for each link"
    )


class WebSearchTool(Tool):
    provider: SearchProvider = SearchProvider.TAVILY
