import os
from typing import Optional

from langchain_core.tools import tool
from redis.asyncio import Redis
from redis.commands.search.query import Query

from src.services.logger.logger_service import LoggerService

logger = LoggerService().setup_logger()


def _build_county_city_field_string(labels):
    """Build formatted string of counties with cities and their fields."""
    county_data = {}

    for label in labels:
        county = label.get("county")
        city = label.get("city")
        field = label.get("field")

        if county and city and field:
            if county not in county_data:
                county_data[county] = {}
            if city not in county_data[county]:
                county_data[county][city] = set()
            county_data[county][city].add(field)

    result_parts = []
    for county, cities in county_data.items():
        city_parts = []
        for city, fields in cities.items():
            fields_str = ", ".join(sorted(fields))
            city_parts.append(f"{city} ({fields_str})")

        cities_str = ", ".join(city_parts)
        result_parts.append(f"{county}: {cities_str}")

    return ", ".join(result_parts)


@tool
async def get_labels_tool(
    index_name: str = "positions",
    job_type: Optional[str] = None,
    app_id: Optional[str | int] = None,
):
    """Get all unique labels organized by counties, cities and fields in formatted string."""

    redis = Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT")),
        username=os.getenv("REDIS_USER"),
        password=os.getenv("REDIS_PASSWORD"),
        decode_responses=True,
    )

    query_parts = []

    if job_type:
        query_parts.append(f"@labels_job_type:{job_type}")
    if app_id is not None:
        query_parts.append(f"@application_id:{app_id}")

    query_str = " ".join(query_parts) if query_parts else "*"
    query = (
        Query(query_str)
        .return_fields(
            "labels_city", "labels_field", "labels_job_type", "labels_county"
        )
        .paging(0, 10000)
    )

    logger.debug(f"[get_labels_tool] Query: {query.query_string}")
    results = await redis.ft(index_name).search(query)

    docs = []
    for doc in results.docs:
        doc_dict = {
            "county": getattr(doc, "labels_county", None),
            "city": getattr(doc, "labels_city", None),
            "field": getattr(doc, "labels_field", None),
        }
        docs.append(doc_dict)

    formatted_string = _build_county_city_field_string(docs)

    await redis.close()

    return formatted_string
