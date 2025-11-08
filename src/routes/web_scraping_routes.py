import json

from fastapi import APIRouter, HTTPException

from src.schemas.web_scraping_schema import (
    WebScrapingRequestSchema,
    WebScrapingResponseSchema,
)
from src.services.logger.logger_service import LoggerService
from src.services.web_scraper.scrapy_web_scraping_service import ScrapySpider

logger = LoggerService().setup_logger()

router = APIRouter(tags=["web_scraping"])


def get_web_scraping_service():
    return ScrapySpider()


@router.post("/api/web-scraping", response_model=WebScrapingResponseSchema)
async def scrape_websites(request: WebScrapingRequestSchema):
    "Extract and process website content automatically."
    try:
        scraping_service = get_web_scraping_service()
        logger.debug(
            f"Received web scraping request with parameters: {request.model_dump()}"
        )

        success, message, scraped_urls, failed_urls, content = (
            await scraping_service.scrape_websites(
                urls=request.urls,
                max_depth=request.max_depth,
                output_type=request.output_type.value,
                output_path=request.output_path,
                vector_db_index=request.vector_db_index or "",
                allowed_domains=request.allowed_domains,
                content_selectors=request.content_selectors,
                excluded_selectors=request.excluded_selectors,
                embedding_model_config=request.embedding_model,
            )
        )

        logger.debug(
            f"Scraping completed. Success: {success}, Message: {message}, Scraped URLs: {scraped_urls}, Failed URLs: {failed_urls}"
        )

        if isinstance(content, (list, dict)):
            content = json.dumps(content)
        return WebScrapingResponseSchema(
            success=success,
            message=message,
            scraped_urls=scraped_urls,
            failed_urls=failed_urls,
            content=content,
        )

    except ConnectionError as ex:
        logger.error(f"Connection error in web scraping: {str(ex)}")
        raise HTTPException(
            status_code=503,
            detail=f"Unable to connect to target websites: {str(ex)}",
        )
    except TimeoutError as ex:
        logger.error(f"Timeout error in web scraping: {str(ex)}")
        raise HTTPException(
            status_code=408,
            detail=f"Web scraping request timed out: {str(ex)}",
        )
    except ValueError as ex:
        logger.error(f"Invalid input in web scraping: {str(ex)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid web scraping parameters: {str(ex)}",
        )
    except Exception as ex:
        logger.error(f"Unexpected error in web scraping: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error scraping websites: {str(ex)}",
        )
