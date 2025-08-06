import os

from langchain_core.tools import tool
from pydantic import Field, create_model
from redis import Redis
from redis.commands.search.query import Query

from src.utils.quote_if_space import quote_if_space


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
        fields[field] = (str, Field(..., description=description))

    return create_model("PositionInput", **fields)


def make_get_position_tool(input_fields):
    position_input_model = make_position_input_model(input_fields)

    def func(**kwargs):
        redis = Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6380")),
            username=os.getenv("REDIS_USER", None),
            password=os.getenv("REDIS_PASSWORD", None),
            decode_responses=True,
        )

        query_parts = []
        label_parts = []

        for field in input_fields:
            value = quote_if_space(kwargs.get(field, ""))
            query_parts.append(f"@{field}:%{value}%")
            label_parts.append(f"%{value}%")

        query_str = " ".join(query_parts)

        if label_parts:
            query_str += f" @labels:({'|'.join(label_parts)})"
        index_name = kwargs.get("index_name", "positions")
        query = Query(query_str)
        results = redis.ft(index_name).search(query)

        return results.docs

    return tool(args_schema=position_input_model)(func)
