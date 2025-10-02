from pydantic import BaseModel

from .model_schema import Model


class TopicValidationRequestSchema(BaseModel):
    question: str
    model: Model | None = None
    allowed_topics: list[str] | None = None
    invalid_topics: list[str] | None = None
    enabled: bool = True


class TopicValidationResponseSchema(BaseModel):
    is_valid: bool
    topic: str
    reason: str | None = None
