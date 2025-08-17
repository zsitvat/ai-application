import os

from langchain_core.tools import tool
from pydantic import Field, create_model
from redis import Redis
from redis.commands.search.query import Query

from src.services.logger.logger_service import LoggerService
from src.utils.quote_if_space import quote_if_space

logger = LoggerService().get_logger(__name__)


def _filter_document_fields(doc):
    """Filter out unwanted fields from a document."""
    unwanted_fields = {
        "_index_name",
        "application_id",
        "starts_at",
        "expires_at",
        "id",
        "payload",
        "embedding",
    }
    filtered_doc = {}
    for key, value in doc.__dict__.items():
        if key not in unwanted_fields:
            filtered_doc[key] = value
    return filtered_doc


def make_position_input_model(input_fields):
    fields = {}

    fields["index_name"] = (
        str,
        Field("positions", description="Index name (fixed, default: positions)"),
    )

    for field in input_fields:
        if field == "index_name":
            continue
        description = f"{field.capitalize()} (custom field)"
        fields[field] = (str, Field("", description=description))

    return create_model("PositionInput", **fields)


def get_position_tool(input_fields):
    position_input_model = make_position_input_model(input_fields)

    @tool(args_schema=position_input_model)
    def position_search_tool(**kwargs):
        """Search positions in a RedisSearch index using the provided input_fields as filters and return matching documents.

        Parameters:
        - index_name: Name of the RedisSearch index (default: positions)
        - other fields listed in input_fields: Used to build a fuzzy query for each field and labels
            Always use all the fields if there are data for them.s

        Returns:
        - A list of matched documents (RedisSearch results.docs)
        """
        redis = Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT")),
            username=os.getenv("REDIS_USER"),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True,
        )

        query_parts = []

        app_id = kwargs.pop("app_id", None)
        application_id = kwargs.pop("application_id", None)
        if app_id is not None:
            query_parts.append(f"@application_id:{app_id}")
        if application_id is not None:
            query_parts.append(f"@application_id:{application_id}")

        for field in input_fields:
            value = kwargs.get(field, "")
            if value:
                if "-" in value:
                    parts = value.split("-")
                    escaped_parts = [
                        quote_if_space(part.strip()) for part in parts if part.strip()
                    ]
                    exact_queries = [f"@{field}:{part}" for part in escaped_parts]
                    fuzzy_queries = [f"@{field}:*{part}*" for part in escaped_parts]
                    all_queries = exact_queries + fuzzy_queries
                    or_query = " | ".join(all_queries)
                    query_parts.append(f"({or_query})")
                else:
                    escaped_value = quote_if_space(value)
                    exact_query = f"@{field}:{escaped_value}"
                    fuzzy_query = f"@{field}:*{escaped_value}*"
                    query_parts.append(f"({exact_query} | {fuzzy_query})")

        query_str = " ".join(query_parts)
        index_name = kwargs.get("index_name", "positions")

        logger.debug(f"[get_position_tool] Position search query: {query_str}")
        logger.debug(f"[get_position_tool] Search kwargs: {kwargs}")

        query = Query(query_str).paging(0, 100)
        results = redis.ft(index_name).search(query)

        # If no results found and job_type was included, try again without job_type
        if results.total == 0 and any(
            "labels_job_type" in part for part in query_parts
        ):

            fallback_query_parts = [
                part for part in query_parts if "labels_job_type" not in part
            ]

            if fallback_query_parts:
                fallback_query_str = " ".join(fallback_query_parts)
                logger.debug(
                    f"[get_position_tool] No results found, trying fallback without job_type: {fallback_query_str}"
                )
                fallback_query = Query(fallback_query_str).paging(0, 100)
                results = redis.ft(index_name).search(fallback_query)

        return [getattr(doc, "text", "") for doc in results.docs]

    return position_search_tool
