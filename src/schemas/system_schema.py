from pydantic import BaseModel
from datetime import datetime


class LogEntry(BaseModel):
    timestamp: datetime
    level: str
    message: str
    source: str
    details: dict | None = None


class LogsRequestSchema(BaseModel):
    level: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    source: str | None = None
    limit: int | None = 100


class LogsResponseSchema(BaseModel):
    logs: list[LogEntry]
    total_count: int
    page: int
    page_size: int


class HealthCheckResponseSchema(BaseModel):
    status: str
    timestamp: str
    response_time_ms: float
    version: str | None = None
    system: dict | None = None
    environment: dict | None = None
    services: dict | None = None
    error: str | None = None
