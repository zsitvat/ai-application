import logging

from fastapi import APIRouter, Depends, HTTPException

from src.schemas.web_scraping_schema import (
    WebScrapingRequestSchema,
    WebScrapingResponseSchema,
)
from src.services.web_scraper.scrapy_web_scraping_service import ScrapySpider

router = APIRouter(tags=["web_scraping"])


def get_web_scraping_service():
    return ScrapySpider()


@router.post("/api/web-scraping", response_model=WebScrapingResponseSchema)
async def scrape_websites(
    request: WebScrapingRequestSchema,
    scraping_service: ScrapySpider = Depends(get_web_scraping_service),
):
    "Extract and process website content automatically."

    try:
        logging.getLogger("logger").debug(
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

        logging.getLogger("logger").debug(
            f"Scraping completed. Success: {success}, Message: {message}, Scraped URLs: {scraped_urls}, Failed URLs: {failed_urls}"
        )

        # Ensure content is a string for response schema
        import json

        if isinstance(content, (list, dict)):
            content = json.dumps(content)
        return WebScrapingResponseSchema(
            success=success,
            message=message,
            scraped_urls=scraped_urls,
            failed_urls=failed_urls,
            content=content,
        )

    except Exception as ex:
        logging.getLogger("logger").error(f"Error in web scraping: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error scraping websites: {str(ex)}",
        )
