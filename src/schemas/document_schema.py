from pydantic import BaseModel

from schemas.graph_schema import Model


class DocumentIngestRequestSchema(BaseModel):
    vector_db_index: str
    files: list[str] | str
    chunk_size: int | None = 1000
    chunk_overlap: int | None = 200
    embedding_model: Model
    index_schema: list[dict] | None = None


class DocumentIngestResponseSchema(BaseModel):
    success: bool
    processed_files: list[str]
    failed_files: list[str]


class DocumentDeleteResponseSchema(BaseModel):
    success: bool
    message: str
    deleted_count: int
