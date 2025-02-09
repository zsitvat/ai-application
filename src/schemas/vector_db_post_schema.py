from pydantic import BaseModel
from enum import Enum

from schemas.model_schema import ModelSchema


class VectorDbPostSchema(BaseModel):
    db_path: str = "./vector_db"
    db_type: str = "deeplake"
    overwrite: bool = False
    sheet_name: str | None = None
    file_load_encoding: str = "utf-8"
    chunk_size: int = 2000
    chunk_overlap: int = 100
    documents: list[str]
    model: ModelSchema = ModelSchema(
        name="text-embedding-3-small",
        type="embedding",
        provider="openai",
        temperature=0,
    )
