import os
from typing import Optional

from langchain_core.tools import tool
from redis.asyncio import Redis
from redis.commands.search.query import Query


@tool
async def get_labels_tool(
    index_name: str = "positions", job_type: Optional[str] = None, **kwargs
) -> dict:
    """
    Get all unique labels from the vector database index.
    Optionally filter by job type (e.g., 'physical', 'intellectual', 'retired',  'student').

    Args:
        index_name (str): Redis index name (default: "positions")
        job_type (str): Filter by job type
    """

    redis = Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT")),
        username=os.getenv("REDIS_USER"),
        password=os.getenv("REDIS_PASSWORD"),
        decode_responses=True,
    )

    if job_type:
        query = (
            Query(f"@labels_job_type:{job_type}")
            .return_fields(
                "labels_city", "labels_field", "labels_job_type", "labels_county"
            )
            .paging(0, 10000)
        )
    else:
        query = (
            Query("*")
            .return_fields(
                "labels_city", "labels_field", "labels_job_type", "labels_county"
            )
            .paging(0, 10000)
        )

    results = await redis.ft(index_name).search(query)
    county_city = {}
    county_field = {}
    job_type_set = set()
    for doc in results.docs:

        city = getattr(doc, "labels_city", None)
        field = getattr(doc, "labels_field", None)
        doc_job_type = getattr(doc, "labels_job_type", None)
        county = getattr(doc, "labels_county", None)

        if county and city:
            if county not in county_city:
                county_city[county] = set()
            county_city[county].add(city)

        if county and field:
            if county not in county_field:
                county_field[county] = set()
            county_field[county].add(field)

        if doc_job_type:
            job_type_set.add(doc_job_type)

    counties_with_cities = {k: list(v) for k, v in county_city.items()}
    counties_with_fields = {k: list(v) for k, v in county_field.items()}

    await redis.close()

    return {
        "county_city": counties_with_cities,
        "county_field": counties_with_fields,
        "job_type": list(job_type_set),
    }
