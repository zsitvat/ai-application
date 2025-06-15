from pydantic import BaseModel
from typing import Optional, Any
from uuid import uuid4, UUID
from enum import Enum


class ApplicationIdentifierSchema(BaseModel):
    tenantIdentifier: int
    applicationIdentifier: int


class RestOperationPostSchema(BaseModel):
    uuid: UUID | str
    applicationIdentifier: ApplicationIdentifierSchema
    platform: str
    userInput: str
    context: Optional[dict[str, str]] = None
    parameters: Optional[dict[str, Any]] = None


class ModelProviderType(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"


class ModelType(str, Enum):
    EMBEDDING = "embedding"
    CHAT = "chat"
    GENERAL = "general"


class Model(BaseModel):
    provider: ModelProviderType = ModelProviderType.OPENAI
    deployment: str | None = None
    name: str | None = "gpt-4o-mini"
    type: ModelType = ModelType.CHAT


class Chain(BaseModel):
    model: Model
    prompt: str
    debug: bool = False


class Embedding(BaseModel):
    model: Model
    index_name: str | None = None
    index_schema: dict[str, Any] | None = None
    search_type: str = "mmr"
    search_kwargs: dict[str, Any] = {"k": 4, "lambda_mult": 0.5}
    max_num_tokens: int = -1
    token_encoding_name: str = "cl100k_base"
    sorted: bool = False
    sort_by: str | None = None


class Agent(BaseModel):
    enabled: bool = True
    chain: Chain
    tool_choice: str = "auto"
    tools: dict[str, dict] = {}


class GraphConfig(BaseModel):
    agents: dict[str, Agent]
    supervisor: Agent
    exception_chain: Agent | None = None
    max_input_length: int = -1
    chat_memory_db: int | None = None
    chat_memory_index_name: str = "chat_memory"
    chat_memory_type: str = "redis"
    chat_history_max_length: int = 12
    tags: list[str] = []
