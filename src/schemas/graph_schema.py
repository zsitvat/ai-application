from pydantic import BaseModel
from uuid import uuid4, UUID


class GraphRequestSchema(BaseModel):
    user_input: str
    user_id: UUID = uuid4()
    context: dict | None = None
    chat_history: list | None = None


class GraphResponseSchema(BaseModel):
    result: str
