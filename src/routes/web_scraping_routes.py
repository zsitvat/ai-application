from fastapi import APIRouter, HTTPException, Depends
import logging

from schemas.web_scraping_schema import (
    WebScrapingRequestSchema,
    WebScrapingResponseSchema,
)
from services.web_scraper.web_scraping_service import WebScrapingService

router = APIRouter()


def get_web_scraping_service():
    return WebScrapingService()


@router.post("/api/web-scraping", response_model=WebScrapingResponseSchema)
async def scrape_websites(
    request: WebScrapingRequestSchema,
    scraping_service: WebScrapingService = Depends(get_web_scraping_service),
):
    "Extract and process website content automatically."

    try:
        success, message, scraped_urls, failed_urls = (
            await scraping_service.scrape_websites(
                urls=request.urls,
                max_depth=request.max_depth,
                output_type=request.output_type.value,
                vector_db_index=request.vector_db_index,
            )
        )

        return WebScrapingResponseSchema(
            success=success,
            message=message,
            scraped_urls=scraped_urls,
            failed_urls=failed_urls,
        )

    except Exception as ex:
        logging.getLogger("logger").error(f"Error in web scraping: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error scraping websites: {str(ex)}",
        )
