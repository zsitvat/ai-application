from pydantic import BaseModel
from enum import Enum


class OutputType(str, Enum):
    DOCX = "docx"
    DOC = "doc"
    PDF = "pdf"
    HTML = "html"
    TEXT = "text"
    VECTOR_DB = "vector_db"


class WebScrapingRequestSchema(BaseModel):
    urls: list[str]
    max_depth: int = 1
    output_type: OutputType = OutputType.TEXT
    vector_db_index: str | None = None


class WebScrapingResponseSchema(BaseModel):
    success: bool
