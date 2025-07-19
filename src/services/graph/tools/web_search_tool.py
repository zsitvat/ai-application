import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import BingSearchAPIWrapper, SerpAPIWrapper
from langchain_core.tools import BaseTool
from langchain_google_community import GoogleSearchAPIWrapper
from pydantic import BaseModel, Field

from schemas.tool_schema import SearchProvider, WebSearchToolInput


class WebSearchTool(BaseTool):
    """Configurable web search tool that supports Google, Bing, SerpAPI, and Tavily search providers."""

    name: str = "web_search"
    description: str = (
        "Search the web for current information using Google, Bing, SerpAPI, or Tavily"
    )
    args_schema: type[BaseModel] = WebSearchToolInput

    provider: SearchProvider = Field(
        default=SearchProvider.TAVILY, description="Search provider to use"
    )
    k: int = Field(default=5, description="Number of search results to return")

    google_api_key: str | None = Field(default=None, description="Google API key")
    google_cse_id: str | None = Field(
        default=None, description="Google Custom Search Engine ID"
    )

    bing_subscription_key: str | None = Field(
        default=None, description="Bing Search subscription key"
    )
    bing_search_url: str = Field(
        default_factory=lambda: os.getenv(
            "BING_SEARCH_URL", "https://api.bing.microsoft.com/v7.0/search"
        ),
        description="Bing Search URL",
    )

    serpapi_api_key: str | None = Field(default=None, description="SerpAPI API key")
    serpapi_params: dict[str, any] = Field(
        default_factory=dict, description="Additional SerpAPI parameters"
    )

    tavily_api_key: str | None = Field(default=None, description="Tavily API key")
    tavily_topic: str = Field(default="general", description="Tavily search topic")
    tavily_search_depth: str = Field(default="basic", description="Tavily search depth")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._search_wrapper = None
        self._setup_search_wrapper()

    def _setup_search_wrapper(self):
        """Setup the appropriate search wrapper based on provider."""
        if self.provider == SearchProvider.GOOGLE:
            self._setup_google_search()
        elif self.provider == SearchProvider.BING:
            self._setup_bing_search()
        elif self.provider == SearchProvider.SERPAPI:
            self._setup_serpapi_search()
        elif self.provider == SearchProvider.TAVILY:
            self._setup_tavily_search()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _setup_google_search(self):
        """Setup Google Search wrapper."""
        api_key = self.google_api_key or os.getenv("GOOGLE_API_KEY")
        cse_id = self.google_cse_id or os.getenv("GOOGLE_CSE_ID")

        if not api_key or not cse_id:
            raise ValueError(
                "Google search requires GOOGLE_API_KEY and GOOGLE_CSE_ID. "
                "Set them as environment variables or pass them to the tool."
            )

        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["GOOGLE_CSE_ID"] = cse_id

        self._search_wrapper = GoogleSearchAPIWrapper(k=self.k)

    def _setup_bing_search(self):
        """Setup Bing Search wrapper."""
        subscription_key = self.bing_subscription_key or os.getenv(
            "BING_SUBSCRIPTION_KEY"
        )
        search_url = self.bing_search_url or os.getenv(
            "BING_SEARCH_URL", "https://api.bing.microsoft.com/v7.0/search"
        )

        if not subscription_key:
            raise ValueError(
                "Bing search requires BING_SUBSCRIPTION_KEY. "
                "Set it as environment variable or pass it to the tool."
            )

        os.environ["BING_SUBSCRIPTION_KEY"] = subscription_key
        os.environ["BING_SEARCH_URL"] = search_url

        self._search_wrapper = BingSearchAPIWrapper(k=self.k)

    def _setup_serpapi_search(self):
        """Setup SerpAPI Search wrapper."""
        api_key = self.serpapi_api_key or os.getenv("SERPAPI_API_KEY")

        if not api_key:
            raise ValueError(
                "SerpAPI search requires SERPAPI_API_KEY. "
                "Set it as environment variable or pass it to the tool."
            )

        os.environ["SERPAPI_API_KEY"] = api_key

        default_params = {
            "engine": "google",
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
        }

        params = {**default_params, **self.serpapi_params}
        self._search_wrapper = SerpAPIWrapper(params=params)

    def _setup_tavily_search(self):
        """Setup Tavily Search wrapper."""
        api_key = self.tavily_api_key or os.getenv("TAVILY_API_KEY")

        if not api_key:
            raise ValueError(
                "Tavily search requires TAVILY_API_KEY. "
                "Set it as environment variable or pass it to the tool."
            )

        os.environ["TAVILY_API_KEY"] = api_key

        self._search_wrapper = TavilySearchResults(
            max_results=self.k,
            search_depth=self.tavily_search_depth,
            topic=self.tavily_topic,
        )

    def _run(self, query: str) -> str:
        """Execute search and return formatted results."""
        try:
            if self.provider == SearchProvider.TAVILY:
                results = self._search_wrapper.invoke({"query": query})
                return self._format_tavily_results(results, query)
            else:
                if hasattr(self._search_wrapper, "results"):
                    results = self._search_wrapper.results(query, self.k)
                else:
                    result_text = self._search_wrapper.run(query)
                    return (
                        f"Search results from SerpAPI for '{query}':\n\n{result_text}"
                    )

                return self._format_results(results, query, self.provider.title())

        except Exception as e:
            return f"Search failed: {str(e)}"

    def _format_results(self, results: list, query: str, provider: str) -> str:
        """Format search results into a readable string."""
        if not results:
            return f"No search results found for query: '{query}'"

        formatted_results = [f"Search results from {provider} for '{query}':\n"]

        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            link = result.get("link", result.get("url", "No link"))
            snippet = result.get("snippet", result.get("content", "No description"))

            formatted_results.append(
                f"{i}. **{title}**\n" f"   URL: {link}\n" f"   Description: {snippet}\n"
            )

        return "\n".join(formatted_results)

    def _format_tavily_results(self, results: list, query: str) -> str:
        """Format Tavily search results into a readable string."""
        if not results:
            return f"No search results found for query: '{query}'"

        formatted_results = [f"Search results from Tavily for '{query}':\n"]

        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            content = result.get("content", "No content")

            formatted_results.append(
                f"{i}. **{title}**\n" f"   URL: {url}\n" f"   Content: {content}\n"
            )

        return "\n".join(formatted_results)
