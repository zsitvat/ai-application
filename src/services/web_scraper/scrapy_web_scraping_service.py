import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from scrapy.crawler import CrawlerRunner
from scrapy.http import Request
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Spider
from scrapy.utils.project import get_project_settings

# Install asyncio reactor before any Twisted-using modules are imported
if "twisted.internet.reactor" not in sys.modules:
    import twisted.internet.asyncioreactor

    twisted.internet.asyncioreactor.install()


class ScrapySpider(Spider):
    """
    Scrapy spider that respects domain restrictions and depth limits.
    """

    name = "spider"

    def __init__(
        self,
        start_urls=None,
        allowed_domains=None,
        max_depth=1,
        service_ref=None,
        *args,
        **kwargs,
    ):
        super(ScrapySpider, self).__init__(*args, **kwargs)

        self.start_urls = start_urls or []
        self.allowed_domains = allowed_domains or []
        self.max_depth = max_depth
        self.scraped_data = {}
        self.failed_urls = []
        self.service_ref = service_ref
        # Logger is a property of the spider, so instead of using `self.logger`, we use a custom logger
        self._custom_logger = logging.getLogger(__name__)

    def start_requests(self):
        """Generate initial requests."""
        self._custom_logger.info(f"Start URLs: {self.start_urls}")
        self._custom_logger.info(f"Allowed domains: {self.allowed_domains}")

        if not self.start_urls:
            self._custom_logger.error("No start URLs provided. Exiting.")
            return

        for url in self.start_urls:
            self._custom_logger.debug(f"Creating request for: {url}")
            yield Request(url, self.parse, errback=self.handle_request_error)

    def parse(self, response):
        """Main parse method - simplified like working example."""
        self._custom_logger.debug(
            f"Parsing: {response.url} (status: {response.status})"
        )

        content = self._extract_content(response)
        self.scraped_data[response.url] = content

        if self.service_ref:
            self.service_ref._current_spider_data = self.scraped_data.copy()

        self._custom_logger.debug(
            f"Extracted {len(content)} characters from {response.url}"
        )

        if self.max_depth > 0:
            link_extractor = LinkExtractor(
                allow_domains=self.allowed_domains,
                deny_extensions=[
                    "jpg",
                    "jpeg",
                    "png",
                    "gif",
                    "pdf",
                    "doc",
                    "docx",
                    "zip",
                    "exe",
                ],
            )

            links = link_extractor.extract_links(response)
            self._custom_logger.debug(f"Found {len(links)} links on {response.url}")

            for link in links:
                self._custom_logger.debug(f"Following link: {link.url}")
                yield Request(
                    url=link.url,
                    callback=self.parse_page,
                    meta={"depth": 1},
                    errback=self.handle_request_error,
                )

    def parse_page(self, response):
        """Parse a single page and extract content."""
        current_depth = response.meta.get("depth", 0)

        self._custom_logger.debug(
            f"Processing page: {response.url} (status: {response.status}, depth: {current_depth})"
        )

        content = self._extract_content(response)
        self.scraped_data[response.url] = content

        self._custom_logger.debug(
            f"Extracted {len(content)} characters from {response.url}"
        )

        if self.service_ref:
            self.service_ref._current_spider_data = self.scraped_data.copy()

        if current_depth < self.max_depth:
            link_extractor = LinkExtractor(
                allow_domains=self.allowed_domains,
                deny_extensions=[
                    "jpg",
                    "jpeg",
                    "png",
                    "gif",
                    "pdf",
                    "doc",
                    "docx",
                    "zip",
                    "exe",
                ],
            )

            links = link_extractor.extract_links(response)
            self._custom_logger.debug(f"Found {len(links)} links on {response.url}")

            for link in links:
                self._custom_logger.debug(
                    f"Collecting link: {link.url} (depth: {current_depth + 1})"
                )
                yield Request(
                    url=link.url,
                    callback=self.parse_page,
                    meta={"depth": current_depth + 1},
                    errback=self.handle_request_error,
                )

    def handle_request_error(self, failure):
        """Handle request failures."""
        self.failed_urls.append(failure.request.url)
        self._custom_logger.error(f"Request failed for URL: {failure.request.url}")

        if self.service_ref:
            self.service_ref._current_spider_failed = self.failed_urls.copy()

    def _extract_content(self, response):
        """Extract readable text content from response."""
        try:
            text = response.css("::text").getall()

            content = " ".join(text)

            lines = (line.strip() for line in content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = " ".join(chunk for chunk in chunks if chunk)

            return content
        except Exception as e:
            self._custom_logger.error(
                f"Error extracting content from {response.url}: {str(e)}"
            )
            return ""


class ScrapyWebScrapingService:
    """
    Scrapy-based web scraping service with domain restrictions and depth control.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scraped_data = {}
        self.failed_urls = []

    def _extract_domains_from_urls(self, urls: list[str]) -> list[str]:
        """Extract domains from URLs."""
        domains = []
        for url in urls:
            try:
                domain = urlparse(url).netloc
                if domain and domain not in domains:
                    domains.append(domain)
            except Exception:
                continue
        return domains

    async def scrape_websites(
        self,
        urls: list[str],
        max_depth: int = 1,
        output_type: str = "text",
        output_path: Optional[str] = None,
        vector_db_index: Optional[str] = None,
        allowed_domains: Optional[list[str]] = None,
    ) -> tuple[bool, str, list[str], list[str], str | None]:
        """
        Scrape websites using Scrapy with domain restriction.

        Args:
            urls: Starting URLs to scrape
            max_depth: Maximum depth for following links
            allowed_domains: list of allowed domains to crawl

        Returns:
            tuple: (success, message, scraped_urls, failed_urls)
        """
        try:

            allowed_domains = self._extract_domains_from_urls(urls)

            self.logger.info(
                f"Starting Scrapy crawl with max_depth={max_depth}, output_type={output_type}, "
                f"vector_db_index={vector_db_index}, allowed_domains={allowed_domains}"
            )

            settings = get_project_settings()
            settings.update(
                {
                    "DOWNLOAD_DELAY": 0.1,
                    "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "LOG_LEVEL": "DEBUG",
                    "RETRY_TIMES": 5,
                }
            )

            self.logger.debug(f"Scrapy settings configured. URLs to crawl: {urls}")
            self.logger.debug(f"Allowed domains: {allowed_domains}")

            runner = CrawlerRunner(settings)

            self._current_spider_data = {}
            self._current_spider_failed = []

            deferred = runner.crawl(
                ScrapySpider,
                start_urls=urls,
                allowed_domains=allowed_domains,
                max_depth=max_depth,
                service_ref=self,
            )

            self.logger.info(f"Spider crawl initiated. Deferred: {type(deferred)}")

            def _extract_results(result):
                self.logger.debug(f"Spider finished with result: {result}")
                self.scraped_data = getattr(self, "_current_spider_data", {})
                self.failed_urls = getattr(self, "_current_spider_failed", [])
                return result

            deferred.addBoth(_extract_results)

            await self._run_scrapy_async(deferred)

            scraped_urls = list(self.scraped_data.keys())
            success = len(scraped_urls) > 0
            message = f"Successfully scraped {len(scraped_urls)} URLs, {len(self.failed_urls)} failed"

            if success and output_path and output_type not in ["vector_db", "string"]:
                try:
                    saved_file = self._save_scraped_content_to_file(
                        output_type, output_path
                    )
                    message += f". Content saved to: {saved_file}"
                    self.logger.info(f"Content saved to: {saved_file}")
                    content_string = f"Content successfully saved to file: {saved_file}"
                except Exception as e:
                    self.logger.error(f"Failed to save content to file: {str(e)}")
                    message += f". Warning: Failed to save content to file: {str(e)}"
                    content_string = f"Error: Failed to save content to file: {str(e)}"
            else:
                content_string = (
                    self.get_scraped_content_as_string()
                    if output_type == "string"
                    else None
                )

            self.logger.info(f"=== SCRAPY CRAWLING COMPLETED ===")
            self.logger.info(f"Results: {message}")
            return success, message, scraped_urls, self.failed_urls, content_string

        except Exception as e:
            error_msg = f"Error during Scrapy web scraping: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg, [], [], None

    async def _run_scrapy_async(self, deferred):
        """Convert Twisted deferred to asyncio."""
        try:
            future = asyncio.Future()

            def callback(result):
                if not future.done():
                    future.set_result(result)
                return result

            def errorback(failure):
                if not future.done():

                    if hasattr(failure.value, "__class__"):
                        error_class = failure.value.__class__.__name__
                        if error_class in ["ConnectionDone", "AlreadyNegotiating"]:
                            self.logger.debug(
                                f"Twisted connection issue (expected): {failure.value}"
                            )
                            future.set_result(None)
                        else:
                            self.logger.error(f"Twisted error: {failure.value}")
                            future.set_exception(failure.value)
                    else:
                        future.set_exception(failure.value)
                return failure

            deferred.addCallbacks(callback, errorback)

            result = await future
            return result

        except Exception as e:
            self.logger.error(f"Error in _run_scrapy_async: {str(e)}")
            raise e

    def _get_scraped_content(self, url: str) -> str:
        """Get scraped content for a specific URL."""
        return self.scraped_data.get(url, "")

    def _get_all_scraped_content(self) -> dict[str, str]:
        """Get all scraped content."""
        return self.scraped_data.copy()

    def _clear_cache(self):
        """Clear all cached data."""
        self.scraped_data.clear()
        self.failed_urls.clear()

    def _save_scraped_content_to_file(self, output_type: str, output_path: str) -> str:
        """Save scraped content to file based on output type."""
        if not self.scraped_data:
            return ""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_content = self._get_all_content()

        if output_type == "text":
            filename = self._save_as_text(output_path, timestamp, all_content)
        elif output_type == "html":
            filename = self._save_as_html(output_path, timestamp)
        elif output_type == "docx":
            filename = self._save_as_docx(output_path, timestamp)
        elif output_type == "pdf":
            filename = self._save_as_pdf(output_path, timestamp)
        else:
            filename = self._save_as_text(output_path, timestamp, all_content)

        return filename

    def _get_all_content(self) -> str:
        all_content = ""
        for url, content in self.scraped_data.items():
            all_content += f"URL: {url}\n\n{content}\n\n{'-'*80}\n\n"
        return all_content

    def _save_as_text(self, output_path, timestamp, all_content):
        filename = f"{output_path}/scraped_content_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(all_content)
        return filename

    def _save_as_html(self, output_path, timestamp):
        filename = f"{output_path}/scraped_content_{timestamp}.html"
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Scraped Content</title>
</head>
<body>
    <h1>Scraped Content - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</h1>
"""
        for url, content in self.scraped_data.items():
            html_content += f"""
    <div style="margin-bottom: 30px; border-bottom: 1px solid #ccc; padding-bottom: 20px;">
        <h2><a href="{url}" target="_blank">{url}</a></h2>
        <pre style="white-space: pre-wrap; font-family: Arial, sans-serif;">{content}</pre>
    </div>
"""
        html_content += "</body></html>"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        return filename

    def _save_as_docx(self, output_path, timestamp):
        """Save scraped content as DOCX file."""
        filename = f"{output_path}/scraped_content_{timestamp}.docx"

        doc = Document()

        doc.add_heading("Scraped Content", 0)
        doc.add_paragraph(
            f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        )
        doc.add_paragraph("")

        for i, (url, content) in enumerate(self.scraped_data.items()):
            if i > 0:
                doc.add_page_break()

            doc.add_heading(f"URL: {url}", level=1)
            doc.add_paragraph("")

            content_lines = content.split("\n")
            for line in content_lines:
                if line.strip():
                    doc.add_paragraph(line.strip())
                else:
                    doc.add_paragraph("")

        doc.save(filename)
        self.logger.debug(f"Successfully saved DOCX file: {filename}")
        return filename

    def _save_as_pdf(self, output_path, timestamp):
        """Save scraped content as PDF file."""
        filename = f"{output_path}/scraped_content_{timestamp}.pdf"

        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=18,
            spaceAfter=30,
            alignment=1,
        )

        url_style = ParagraphStyle(
            "URLHeading",
            parent=styles["Heading2"],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=24,
        )

        story.append(Paragraph("Scraped Content", title_style))
        story.append(
            Paragraph(
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 14))

        for url, content in self.scraped_data.items():
            story.append(Paragraph(f"URL: {url}", url_style))

            content_lines = content.split("\n")
            for line in content_lines:
                if line.strip():
                    escaped_line = (
                        line.strip()
                        .replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    story.append(Paragraph(escaped_line, styles["Normal"]))
                else:
                    story.append(Spacer(1, 7))

            story.append(Spacer(1, 22))

        doc.build(story)
        self.logger.debug(f"Successfully saved PDF file: {filename}")
        return filename

    def get_scraped_content_as_string(self) -> str:
        """Get all scraped content as a single string."""
        return self._get_all_content()
