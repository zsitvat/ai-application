from pydantic import BaseModel
from enum import Enum


class OutputType(str, Enum):
    DOCX = "docx"
    DOC = "doc"
    PDF = "pdf"
    HTML = "html"
    TEXT = "text"
    VECTOR_DB = "vector_db"
    STRING = "string"


class WebScrapingRequestSchema(BaseModel):
    urls: list[str]
    max_depth: int = 1
    output_type: OutputType = OutputType.TEXT
    output_path: str | None = None
    vector_db_index: str | None = None
    allowed_domains: list[str] | None = None


class WebScrapingResponseSchema(BaseModel):
    success: bool
    message: str
    scraped_urls: list[str]
    failed_urls: list[str]
    content: str | None = None
