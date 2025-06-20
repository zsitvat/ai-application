import sys
import json
import os
from pathlib import Path
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from scrapy_web_scraping_service import ScrapySpider

scraped_data = {}


def main():
    global scraped_data
    url = sys.argv[1]
    max_depth = int(sys.argv[2])
    allowed_domains = (
        sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else []
    )

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
    )
    process.start()

    print(json.dumps(scraped_data))


if __name__ == "__main__":
    main()
