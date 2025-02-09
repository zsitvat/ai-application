from pydantic import BaseModel
from typing import Optional
from uuid import uuid4

from schemas.model_schema import ModelSchema
from schemas.tool_schema import Tool, WebSearchTool, RetriverTool


class AgentRequestPostSchema(BaseModel):
    prompt: str
    user_input: str
    user_id: str = str(uuid4())
    model: ModelSchema
    tools: Optional[list[Tool | WebSearchTool | RetriverTool]]
