from datetime import datetime

import pytest

from src.schemas.system_schema import (
    HealthCheckResponseSchema,
    LogEntry,
    LogsRequestSchema,
    LogsResponseSchema,
)


def test_log_entry():
    """Test LogEntry instantiation and field values."""
    obj = LogEntry(
        timestamp="2025-08-20T12:00:00Z",
        level="INFO",
        message="System started",
        source="system",
        details={"ip": "127.0.0.1"},
    )
    assert obj.timestamp == "2025-08-20T12:00:00Z"
    assert obj.level == "INFO"
    assert obj.message == "System started"
    assert obj.source == "system"
    assert obj.details == {"ip": "127.0.0.1"}


def test_logs_request_schema():
    """Test LogsRequestSchema instantiation and field values."""
    obj = LogsRequestSchema(
        level="ERROR",
        start_date=datetime(2025, 8, 20),
        end_date=datetime(2025, 8, 21),
        source="system",
        limit=10,
    )
    assert obj.level == "ERROR"
    assert obj.start_date == datetime(2025, 8, 20)
    assert obj.end_date == datetime(2025, 8, 21)
    assert obj.source == "system"
    assert obj.limit == 10


def test_logs_response_schema():
    """Test LogsResponseSchema instantiation and field values."""
    log = LogEntry(
        timestamp="2025-08-20T12:00:00Z", level="INFO", message="System started"
    )
    obj = LogsResponseSchema(logs=[log], total_count=1, page=1, page_size=10)
    assert obj.logs == [log]
    assert obj.total_count == 1
    assert obj.page == 1
    assert obj.page_size == 10


def test_health_check_response_schema():
    """Test HealthCheckResponseSchema instantiation and field values."""
    obj = HealthCheckResponseSchema(
        status="ok",
        timestamp="2025-08-20T12:00:00Z",
        response_time_ms=123.45,
        version="1.0.0",
        system={"cpu": "Intel"},
        environment={"os": "Linux"},
        services={"db": "up"},
        error=None,
    )
    assert obj.status == "ok"
    assert obj.timestamp == "2025-08-20T12:00:00Z"
    assert obj.response_time_ms == pytest.approx(123.45)
    assert obj.version == "1.0.0"
    assert obj.system == {"cpu": "Intel"}
    assert obj.environment == {"os": "Linux"}
    assert obj.services == {"db": "up"}
    assert obj.error is None
