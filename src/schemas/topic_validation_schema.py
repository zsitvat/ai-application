from pydantic import BaseModel


class TopicValidationRequestSchema(BaseModel):
    question: str


class TopicValidationResponseSchema(BaseModel):
    is_valid: bool
    topic: str
    reason: str | None = None
