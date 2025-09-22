"""
Application constants and configuration values.
"""

# API Configuration
DEFAULT_PORT = 5000
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_RATE_LIMIT = 20
DEFAULT_API_VERSION = "2024-08-01-preview"

# Model Configuration
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 60

# Provider Constants
OPENAI_PROVIDER = "openai"
AZURE_PROVIDER = "azure"
ANTHROPIC_PROVIDER = "anthropic"

# Model Types
CHAT_TYPE = "chat"
COMPLETIONS_TYPE = "completions"
EMBEDDING_TYPE = "embedding"

# Health Check
HEALTH_STATUS_HEALTHY = "healthy"
HEALTH_STATUS_UNHEALTHY = "unhealthy"

# Pagination
DEFAULT_LIMIT = 100
DEFAULT_PAGE = 1

# Cache
DEFAULT_CACHE_TTL = 60

# Recursion
DEFAULT_RECURSION_LIMIT = 1
