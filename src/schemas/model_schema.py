from enum import Enum

from pydantic import BaseModel


class ModelProviderType(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"


class ModelType(str, Enum):
    EMBEDDING = "embedding"
    CHAT = "chat"
    GENERAL = "general"


class Model(BaseModel):
    provider: ModelProviderType = ModelProviderType.OPENAI
    deployment: str | None = None
    name: str | None = "gpt-4o-mini"
    type: ModelType = ModelType.CHAT
