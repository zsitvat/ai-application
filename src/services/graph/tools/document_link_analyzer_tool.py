import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.schemas.tool_schema import DocumentLinkAnalyzerInput
from src.services.logger.logger_service import LoggerService

logger = LoggerService().get_logger(__name__)


class DocumentLinkAnalyzerTool(BaseTool):
    """Tool for analyzing and processing URLs found in text content or documents."""

    name: str = "document_link_analyzer"
    description: str = (
        "Analyze and extract content from URLs found in text or documents. "
        "Automatically detects links and provides summaries of their content."
    )
    args_schema: type[BaseModel] = DocumentLinkAnalyzerInput

    max_content_length: int = Field(
        default=5000, description="Maximum content length to extract per URL"
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")
    user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        description="User agent for web requests",
    )

    def _run(self, text: str, max_links: int = 10, include_summary: bool = True) -> str:
        """Extract and analyze URLs from the given text."""
        try:
            urls = self._extract_urls(text)

            if not urls:
                return "No URLs found in the provided text."

            if len(urls) > max_links:
                urls = urls[:max_links]
                logger.info(f"Limited processing to {max_links} URLs")

            results = []
            results.append(f"Found {len(urls)} URL(s) to analyze:\n")

            for i, url in enumerate(urls, 1):
                try:
                    content_data = self._fetch_url_content(url)
                    if content_data:
                        formatted_result = self._format_url_result(
                            i, url, content_data, include_summary
                        )
                        results.append(formatted_result)
                    else:
                        results.append(
                            f"{i}. **{url}**\n   Status: Failed to fetch content\n"
                        )

                except Exception as e:
                    logger.error(f"Error processing URL {url}: {str(e)}")
                    results.append(f"{i}. **{url}**\n   Error: {str(e)}\n")

            return "\n".join(results)

        except Exception as e:
            return f"Error analyzing links: {str(e)}"

    def _extract_urls(self, text: str) -> list[str]:
        """Extract URLs from text using regex."""
        url_pattern = re.compile(r'https?://[^\s<>"]+')
        urls = url_pattern.findall(text)

        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        return unique_urls

    def _fetch_url_content(self, url: str) -> dict[str, str] | None:
        """Fetch and extract content from a URL."""
        try:
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }

            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            title = soup.find("title")
            title_text = title.get_text().strip() if title else "No title"

            meta_desc = soup.find("meta", attrs={"name": "description"})
            description = meta_desc.get("content", "").strip() if meta_desc else ""

            content = self._extract_main_content(soup)

            if len(content) > self.max_content_length:
                content = content[: self.max_content_length] + "..."

            return {
                "title": title_text,
                "description": description,
                "content": content,
                "url": url,
                "status": "success",
            }

        except requests.RequestException as e:
            logger.error(f"Request error for {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return None

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from HTML."""
        for script in soup(["script", "style"]):
            script.decompose()

        main_selectors = [
            "main",
            "article",
            ".content",
            ".post-content",
            ".entry-content",
            ".article-content",
            "#content",
            ".main-content",
            ".post-body",
        ]

        main_content = None
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.find("body")

        if not main_content:
            return soup.get_text()

        text = main_content.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return text

    def _format_url_result(
        self, index: int, url: str, content_data: dict, include_summary: bool
    ) -> str:
        """Format the result for a single URL."""
        result = [f"{index}. **{content_data['title']}**"]
        result.append(f"   URL: {url}")

        if content_data["description"]:
            result.append(f"   Description: {content_data['description']}")

        if include_summary and content_data["content"]:
            summary = content_data["content"][:200]
            if len(content_data["content"]) > 200:
                summary += "..."
            result.append(f"   Summary: {summary}")

        result.append("")
        return "\n".join(result)

    def _validate_url(self, url: str) -> bool:
        """Validate if the URL is properly formed."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
