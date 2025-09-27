"""
Application configuration management.

This module provides centralized configuration management for the application,
handling environment variables and providing type-safe configuration access.
"""

import os

from pydantic import BaseModel, Field, computed_field, field_validator

from .constants import (
    DEFAULT_API_VERSION,
    DEFAULT_CACHE_TTL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_MAX_FILE_SIZE_MB,
    DEFAULT_MAX_CONTENT_LENGTH,
    DEFAULT_PORT,
    DEFAULT_RATE_LIMIT,
    DEFAULT_RECURSION_LIMIT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
)


class RedisConfig(BaseModel):
    """Redis database configuration."""

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6380, ge=1, le=65535, description="Redis port")
    user: str | None = Field(default=None, description="Redis username")
    password: str | None = Field(default=None, description="Redis password")
    db: str | None = Field(default=None, description="Redis database")
    history_db: str | None = Field(default=None, description="Redis history database")

    @computed_field
    @property
    def url(self) -> str:
        """Get Redis connection URL."""
        auth_part = (
            f"{self.user}:{self.password}@" if self.user and self.password else ""
        )
        db_part = f"/{self.db}" if self.db else ""
        return f"redis://{auth_part}{self.host}:{self.port}{db_part}"

    @computed_field
    @property
    def history_url(self) -> str:
        """Get Redis connection URL for history database."""
        auth_part = (
            f"{self.user}:{self.password}@" if self.user and self.password else ""
        )
        db_part = f"/{self.history_db}" if self.history_db else ""
        return f"redis://{auth_part}{self.host}:{self.port}{db_part}"


class AzureConfig(BaseModel):
    """Azure OpenAI configuration."""

    base_url: str | None = Field(default=None, description="Azure OpenAI base URL")
    api_version: str = Field(
        default=DEFAULT_API_VERSION, description="Azure API version"
    )
    deployment_name: str | None = Field(
        default=None, description="Azure deployment name"
    )
    embedding_deployment_name: str | None = Field(
        default=None, description="Azure embedding deployment name"
    )


class DocumentConfig(BaseModel):
    """Document processing configuration."""

    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE, ge=100, description="Text chunk size for processing"
    )
    chunk_overlap: int = Field(
        default=DEFAULT_CHUNK_OVERLAP, ge=0, description="Overlap between text chunks"
    )
    max_content_length: int = Field(
        default=DEFAULT_MAX_CONTENT_LENGTH, ge=100, description="Maximum content length"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default=DEFAULT_LOG_LEVEL, description="Logging level")
    file_path: str | None = Field(default=None, description="Log file path")
    max_file_size_mb: int = Field(
        default=DEFAULT_LOG_MAX_FILE_SIZE_MB,
        ge=1,
        description="Maximum file size in MB",
    )
    backup_count: int = Field(
        default=DEFAULT_LOG_BACKUP_COUNT, ge=0, description="Number of backup log files"
    )

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()


class TracingConfig(BaseModel):
    """Tracing configuration."""

    tracer_type: str = Field(default="langsmith", description="Tracer type")
    langfuse_secret_key: str | None = Field(
        default=None, description="Langfuse secret key"
    )
    langfuse_public_key: str | None = Field(
        default=None, description="Langfuse public key"
    )
    langfuse_project: str | None = Field(
        default=None, description="Langfuse project ID"
    )
    langfuse_endpoint: str | None = Field(
        default=None, description="Langfuse endpoint URL"
    )

    @field_validator("tracer_type")
    @classmethod
    def validate_tracer_type(cls, v: str) -> str:
        """Validate tracer type."""
        allowed_types = {"langsmith", "langfuse", "none"}
        if v.lower() not in allowed_types:
            raise ValueError(f"Tracer type must be one of: {allowed_types}")
        return v.lower()


class AppConfig(BaseModel):
    """Main application configuration."""

    port: int = Field(
        default=DEFAULT_PORT, ge=1, le=65535, description="Application port"
    )
    rate_limit: int = Field(
        default=DEFAULT_RATE_LIMIT, ge=1, description="Rate limit per minute"
    )
    recursion_limit: int = Field(
        default=DEFAULT_RECURSION_LIMIT, ge=1, description="Graph recursion limit"
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0, description="LLM temperature"
    )
    timeout: int = Field(
        default=DEFAULT_TIMEOUT, ge=1, description="Request timeout in seconds"
    )
    cache_ttl: int = Field(
        default=DEFAULT_CACHE_TTL, ge=0, description="Cache TTL in seconds"
    )

    # Sub-configurations
    redis: RedisConfig = Field(
        default_factory=RedisConfig, description="Redis configuration"
    )
    azure: AzureConfig = Field(
        default_factory=AzureConfig, description="Azure configuration"
    )
    document: DocumentConfig = Field(
        default_factory=DocumentConfig, description="Document processing configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
    tracing: TracingConfig = Field(
        default_factory=TracingConfig, description="Tracing configuration"
    )

    # API Keys and Secrets
    embedding_provider: str | None = Field(
        default=None, description="Embedding provider"
    )
    embedding_model: str | None = Field(
        default=None, description="Embedding model name"
    )
    google_api_key: str | None = Field(default=None, description="Google API key")
    google_cse_id: str | None = Field(default=None, description="Google CSE ID")
    bing_subscription_key: str | None = Field(
        default=None, description="Bing subscription key"
    )
    bing_search_url: str | None = Field(default=None, description="Bing search URL")
    serpapi_api_key: str | None = Field(default=None, description="SerpAPI key")
    tavily_api_key: str | None = Field(default=None, description="Tavily API key")
    data_api_base_url: str | None = Field(default=None, description="Data API base URL")
    issue_tracker_endpoint: str | None = Field(
        default=None, description="Issue tracker endpoint"
    )
    positions_index_schema_path: str | None = Field(
        default=None, description="Positions index schema path"
    )


def load_config() -> AppConfig:
    """Load application configuration from environment variables.

    Returns:
        AppConfig: Loaded application configuration with validation
    """
    return AppConfig(
        # Main application settings
        port=int(os.getenv("PORT", str(DEFAULT_PORT))),
        rate_limit=int(os.getenv("RATELIMIT", str(DEFAULT_RATE_LIMIT))),
        recursion_limit=int(os.getenv("RECURSION_LIMIT", str(DEFAULT_RECURSION_LIMIT))),
        temperature=float(os.getenv("TEMPERATURE", str(DEFAULT_TEMPERATURE))),
        timeout=int(os.getenv("TIMEOUT", str(DEFAULT_TIMEOUT))),
        cache_ttl=int(os.getenv("CACHE_TTL", str(DEFAULT_CACHE_TTL))),
        # Sub-configurations
        redis=RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            user=os.getenv("REDIS_USER"),
            password=os.getenv("REDIS_PASSWORD"),
            db=os.getenv("REDIS_DB"),
            history_db=os.getenv("REDIS_HISTORY_DB"),
        ),
        azure=AzureConfig(
            base_url=os.getenv("AZURE_BASE_URL"),
            api_version=os.getenv("AZURE_API_VERSION", DEFAULT_API_VERSION),
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
            embedding_deployment_name=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),
        ),
        document=DocumentConfig(
            chunk_size=int(os.getenv("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP))),
            max_content_length=int(
                os.getenv("MAX_CONTENT_LENGTH", str(DEFAULT_MAX_CONTENT_LENGTH))
            ),
        ),
        logging=LoggingConfig(
            level=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL),
            file_path=os.getenv("LOG_FILE_PATH"),
            max_file_size_mb=int(
                os.getenv("LOG_MAX_FILE_SIZE_MB", str(DEFAULT_LOG_MAX_FILE_SIZE_MB))
            ),
            backup_count=int(
                os.getenv("LOG_BACKUP_COUNT", str(DEFAULT_LOG_BACKUP_COUNT))
            ),
        ),
        tracing=TracingConfig(
            tracer_type=os.getenv("TRACER_TYPE", "langsmith"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_project=os.getenv("LANGFUSE_PROJECT"),
            langfuse_endpoint=os.getenv("LANGFUSE_ENDPOINT"),
        ),
        # API Keys and Secrets
        embedding_provider=os.getenv("EMBEDDING_PROVIDER"),
        embedding_model=os.getenv("EMBEDDING_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        google_cse_id=os.getenv("GOOGLE_CSE_ID"),
        bing_subscription_key=os.getenv("BING_SUBSCRIPTION_KEY"),
        bing_search_url=os.getenv("BING_SEARCH_URL"),
        serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        data_api_base_url=os.getenv("DATA_API_BASE_URL"),
        issue_tracker_endpoint=os.getenv("ISSUE_TRACKER_ENDPOINT"),
        positions_index_schema_path=os.getenv("POSITIONS_INDEX_SCHEMA_PATH"),
    )


# Global configuration instance
config = load_config()
