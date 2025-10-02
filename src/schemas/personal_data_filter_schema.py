from pydantic import BaseModel

from .model_schema import Model


class PersonalDataFilterRequestSchema(BaseModel):
    text: str
    model: Model | None = None
    sensitive_words: list[str] | None = None
    regex_patterns: list[str] | None = None
    prompt: str | None = None
    mask_char: str = "*"


class PersonalDataFilterResponseSchema(BaseModel):
    filtered_text: str
    original_text: str
