from pydantic import BaseModel

class AgentAnswerResponseSchema(BaseModel):
    answer: str

class VectorDbResponseSchema(BaseModel):
    response: str
