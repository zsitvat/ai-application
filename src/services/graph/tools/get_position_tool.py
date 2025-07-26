from langchain_core.tools import tool
import os

from redis import Redis
from redis.commands.search.query import Query


@tool
def get_position_tool(county, city, field, job_type, index_name="positions"):
    """
    Position search tool that finds jobs by county, city, field, and job type using fuzzy metadata filtering in Redis.

    Args:
        county (str): County
        city (str): City
        field (str): Field
        job_type (str): Job type
        index_name (str): RediSearch index name

    Returns:
        list: Found position documents
    """

    redis = Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        username=os.getenv("REDIS_USER", None),
        password=os.getenv("REDIS_PASSWORD", None),
        decode_responses=True,
    )

    query_str = (
        f"@county:%{county}% @city:%{city}% @field:%{field}% @job_type:%{job_type}% "
        f"@labels:(%{county}%|%{city}%|%{field}%|%{job_type}%)"
    )

    query = Query(query_str)
    results = redis.ft(index_name).search(query)
    return results.docs
