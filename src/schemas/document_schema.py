from typing import Optional

from pydantic import BaseModel, Field

from schemas.model_schema import Model


class DocumentIngestRequestSchema(BaseModel):
    vector_db_index: str
    files: list[str] | str
    chunk_size: int = 8000
    chunk_overlap: int = 300
    json_data: list[dict] | None = None
    embedding_model: Optional[Model] = Field(
        default=None,
        description="Embedding model configuration. If not provided, environment variables will be used.",
        json_schema_extra={
            "example": {
                "provider": "openai",
                "deployment": None,
                "name": "text-embedding-3-large",
                "type": "embedding",
            }
        },
    )
    index_schema: Optional[list[dict]] = Field(
        default=None,
        description="Redis index schema fields definition. If not provided, default schema will be used.",
        json_schema_extra={
            "example": [
                {"name": "content", "type": "text"},
                {"name": "source", "type": "text"},
                {"name": "document_index", "type": "numeric"},
                {
                    "name": "vector",
                    "type": "vector",
                    "attrs": {
                        "dims": 1536,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": "float32",
                    },
                },
            ]
        },
    )


class DocumentIngestResponseSchema(BaseModel):
    success: bool
    processed_files: list[str]
    failed_files: list[str]
    message: str = ""


class DocumentDeleteResponseSchema(BaseModel):
    success: bool
    message: str
    deleted_count: int


class PositionsIngestRequestSchema(BaseModel):
    file_path: str
    index_name: str
