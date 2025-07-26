from enum import Enum
from typing import Annotated, Any
from uuid import UUID, uuid4

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import AliasChoices, BaseModel, Field
from typing_extensions import TypedDict

from .model_schema import Model


class AgentState(TypedDict):
    """State shared between agents in the graph."""

    messages: Annotated[list[AnyMessage], add_messages]
    next: str
    last_agent: str | list | None = None
    user_input: str
    context: dict[str, Any]
    parameters: dict[str, Any]
    user_id: str | UUID = uuid4()


class ApplicationIdentifierSchema(BaseModel):
    tenantIdentifier: int | str
    applicationIdentifier: int | str


class PlatformType(str, Enum):
    GOOGLE_CHAT = "google_chat"
    VIBER = "viber"
    FACEBOOK_MESSENGER = "facebook_messenger"
    LIVECHAT = "livechat"
    WEBCHAT = "webchat"


class RestOperationPostSchema(BaseModel):
    uuid: UUID | str
    applicationIdentifier: ApplicationIdentifierSchema
    platform: PlatformType = PlatformType.WEBCHAT
    user_input: str = Field(validation_alias=AliasChoices("user_input", "userInput"))
    context: dict[str, str] = {}
    parameters: dict[str, Any] | None = None


class CheckpointerType(str, Enum):
    MEMORY = "memory"
    REDIS = "redis"
    DATA = "data"
    CUSTOM = "custom"


class Chain(BaseModel):
    model: Model
    prompt_id: str
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


class TopicValidatorConfig(BaseModel):
    enabled: bool = True
    model: Model
    allowed_topics: list[str] = []
    invalid_topics: list[str] = []


class PersonalDataFilterConfig(BaseModel):
    enabled: bool = True
    chain: Chain
    sensitive_data_types: list[str] = []
    mask_char: str = "*"


class GraphConfig(BaseModel):
    agents: dict[str, Agent]
    supervisor: Agent
    exception_chain: Agent | None = None
    topic_validator: TopicValidatorConfig | None = None
    personal_data_filter: PersonalDataFilterConfig | None = None
    max_input_length: int = -1
    chat_memory_db: int | None = None
    chat_memory_index_name: str = "chat_memory"
    chat_memory_type: str = "redis"
    chat_history_max_length: int = 12
    tags: list[str] = []
    enable_checkpointer: bool = True
    checkpointer_type: CheckpointerType = CheckpointerType.MEMORY
    allow_supervisor_finish: bool = True
    recursion_limit: int = 3
