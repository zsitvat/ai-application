import sys
import json
import base64
from pathlib import Path
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from src.services.web_scraper.scrapy_web_scraping_service import ScrapySpider

scraped_data = {}


def main():
    """Main function for running the Scrapy spider in a subprocess."""
    global scraped_data
    url = sys.argv[1]
    max_depth = int(sys.argv[2])
    allowed_domains = (
        sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else []
    )

    content_selectors = None
    excluded_selectors = None
    if len(sys.argv) > 4 and sys.argv[4]:
        try:
            config_b64 = sys.argv[4]
            config_json = base64.b64decode(config_b64).decode("utf-8")
            selectors_config = json.loads(config_json)

            content_selectors = selectors_config.get("content_selectors")
            excluded_selectors = selectors_config.get("excluded_selectors")
        except Exception as e:
            sys.stderr.write(f"Error parsing selectors configuration: {str(e)}\n")
            sys.exit(1)

    class DataCapturingSpider(ScrapySpider):
        def closed(self, reason):
            global scraped_data
            scraped_data = self.scraped_data.copy()

    process = CrawlerProcess(get_project_settings())

    process.crawl(
        DataCapturingSpider,
        start_urls=[url],
        max_depth=max_depth,
        allowed_domains=allowed_domains,
        content_selectors=content_selectors,
        excluded_selectors=excluded_selectors,
    )
    process.start()

    print(json.dumps(scraped_data))


if __name__ == "__main__":
    main()
