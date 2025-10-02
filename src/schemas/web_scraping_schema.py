from enum import Enum

from pydantic import BaseModel, field_validator

from schemas.model_schema import Model, ModelType


class OutputType(str, Enum):
    DOCX = "docx"
    PDF = "pdf"
    HTML = "html"
    TXT = "txt"
    VECTOR_DB = "vector_db"
    STRING = "string"
    JSON = "json"


class WebScrapingRequestSchema(BaseModel):
    urls: list[str]
    max_depth: int = 1
    output_type: OutputType = OutputType.STRING
    output_path: str | None = None
    vector_db_index: str | None = None
    allowed_domains: list[str] | None = None
    content_selectors: list[str] | None = None
    excluded_selectors: list[str] | None = None
    embedding_model: Model = Model(
        provider="openai",
        name="text-embedding-3-large",
        type="embedding",
    )

    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_type(cls, v):
        if v is not None and v.type != ModelType.EMBEDDING:
            raise ValueError(f"Model type must be 'embedding', got '{v.type}'")
        return v


class WebScrapingResponseSchema(BaseModel):
    success: bool
    message: str
    scraped_urls: list[str]
    failed_urls: list[str]
    content: str | None = None
