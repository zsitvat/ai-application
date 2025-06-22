import sys
import json
import base64
import tempfile
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

    sys.stderr.write(
        f"Starting subprocess with URL: {url}, max_depth: {max_depth}, allowed_domains: {allowed_domains}\n"
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
            sys.stderr.write(
                f"Using custom selectors - content: {content_selectors}, excluded: {excluded_selectors}\n"
            )
        except Exception as e:
            sys.stderr.write(f"Error parsing selectors configuration: {str(e)}\n")
            sys.exit(1)

    class DataCapturingSpider(ScrapySpider):
        def closed(self, reason):
            global scraped_data
            scraped_data = self.scraped_data.copy()
            sys.stderr.write(
                f"Spider closed with reason: {reason}, scraped {len(scraped_data)} pages\n"
            )

    try:
        process = CrawlerProcess(get_project_settings())

        process.crawl(
            DataCapturingSpider,
            start_urls=[url],
            max_depth=max_depth,
            allowed_domains=allowed_domains,
            content_selectors=content_selectors,
            excluded_selectors=excluded_selectors,
        )

        sys.stderr.write("Starting Scrapy process...\n")
        process.start()

        sys.stderr.write(
            f"Scrapy process completed. Final scraped_data: {len(scraped_data)} items\n"
        )

        output_file_path = (
            Path(__file__).parent.parent.parent.parent / "files" / "temp_file.json"
        )

        try:
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            sys.stderr.write(f"Created/verified directory: {output_file_path.parent}\n")
        except Exception as dir_error:
            sys.stderr.write(
                f"Error creating directory {output_file_path.parent}: {dir_error}\n"
            )
            sys.exit(1)

        try:
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                json.dump(scraped_data, output_file, ensure_ascii=False, indent=2)
            sys.stderr.write(
                f"Successfully wrote {len(scraped_data)} items to {output_file_path}\n"
            )
            print(str(output_file_path))
        except Exception as write_error:
            sys.stderr.write(
                f"Error writing to file {output_file_path}: {write_error}\n"
            )
            sys.exit(1)

    except Exception as e:
        sys.stderr.write(f"Error in subprocess: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
