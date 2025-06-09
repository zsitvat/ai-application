from schemas.model_schema import ModelSchema
from schemas.tool_schema import Tool


class RetriverTool(Tool):
    vector_db_path: str
    model: ModelSchema = ModelSchema(
        name="text-embedding-3-large", type="embedding", provider="openai"
    )
    search_kwargs: SearchKwargs = SearchKwargs(
        k=5, threshold=0.5, search_type="similarity"
    )
