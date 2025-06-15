from pydantic import BaseModel

from schemas.graph_schema import Model


class PersonalDataFilterConfigSchema(BaseModel):
    sensitive_words: list[str] | None = None
    regex_patterns: list[str] | None = None
    model: Model | None = None
    prompt: str | None = None


class PersonalDataFilterRequestSchema(BaseModel):
    text: str
    config: PersonalDataFilterConfigSchema | None = None


class PersonalDataFilterResponseSchema(BaseModel):
    filtered_text: str
    original_text: str
