from enum import Enum
from typing import Annotated, Any
from uuid import UUID

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import AliasChoices, BaseModel, Field
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State shared between agents in the graph."""

    messages: Annotated[list[AnyMessage], add_messages]
    next: str
    user_input: str
    context: dict[str, Any]
    parameters: dict[str, Any]


class ApplicationIdentifierSchema(BaseModel):
    tenantIdentifier: int | str
    applicationIdentifier: int | str


class RestOperationPostSchema(BaseModel):
    uuid: UUID | str
    applicationIdentifier: ApplicationIdentifierSchema
    platform: str
    user_input: str = Field(validation_alias=AliasChoices("user_input", "userInput"))
    context: dict[str, str] | None = None
    parameters: dict[str, Any] | None = None


class ModelProviderType(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"


class ModelType(str, Enum):
    EMBEDDING = "embedding"
    CHAT = "chat"
    GENERAL = "general"


class CheckpointerType(str, Enum):
    MEMORY = "memory"
    REDIS = "redis"
    DATA = "data"
    CUSTOM = "custom"


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
    enable_checkpointer: bool = True
    checkpointer_type: CheckpointerType = CheckpointerType.MEMORY
    allow_supervisor_finish: bool = True
