import os

from langchain_core.tools import tool
from redis import Redis
from redis.commands.search.query import Query


@tool
def get_labels_tool(index_name: str = "positions") -> dict:
    """
    Get all unique labels from the vector database index.
    """

    redis = Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6380")),
        username=os.getenv("REDIS_USER", None),
        password=os.getenv("REDIS_PASSWORD", None),
        decode_responses=True,
    )

    query = Query("*=>[RETURN 1 labels]").paging(0, 10000)
    results = redis.ft(index_name).search(query)
    county_city = {}
    field_set = set()
    job_type_set = set()
    for doc in results.docs:
        labels = getattr(doc, "labels", None)
        if labels and isinstance(labels, dict):
            city = labels.get("city")
            field = labels.get("field")
            job_type = labels.get("job_type")
            county = labels.get("county")
            if county and city:
                if county not in county_city:
                    county_city[county] = set()
                county_city[county].add(city)
            if field:
                field_set.add(field)
            if job_type:
                job_type_set.add(job_type)

    county_city_out = {k: list(v) for k, v in county_city.items()}
    return {
        "county_city": county_city_out,
        "field": list(field_set),
        "job_type": list(job_type_set),
    }
