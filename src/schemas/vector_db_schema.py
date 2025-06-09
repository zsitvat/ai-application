from pydantic import BaseModel
from enum import Enum

from schemas.model_schema import ModelSchema


class VectorDbPostSchema(BaseModel):
    overwrite: bool = False
    file_load_encoding: str = "utf-8"
    chunk_size: int = 5000
    chunk_overlap: int = 250
    documents: list[str]


class DeeplakeVectorDbPostSchema(VectorDbPostSchema):
    db_path: str = "./vector_db"


class RedisVectorDbPostSchema(VectorDbPostSchema):
    db_path: str = "redis://localhost:6380"


class VectorDbResponseSchema(BaseModel):
    success: bool
