from schemas.graph_schema import Model
from schemas.tool_schema import SearchKwargs, Tool


class RetriverTool(Tool):
    vector_db_path: str
    model: Model = Model(
        name="text-embedding-3-large", type="embedding", provider="openai"
    )
    search_kwargs: SearchKwargs = SearchKwargs(
        k=5, threshold=0.5, search_type="similarity"
    )


class WebSearchTool(Tool):
    pass
