from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.app import create_app


@pytest.fixture(autouse=True)
def mock_topic_validator_service(monkeypatch):
    mock_service = MagicMock()
    mock_service.validate_topic = AsyncMock(
        return_value=(True, "general", "Valid topic")
    )
    monkeypatch.setattr(
        "src.routes.topic_validation_routes.get_topic_validator_service",
        lambda: mock_service,
    )
    return mock_service


def test_topic_validation_route_exists(mock_topic_validator_service):
    client = TestClient(create_app())
    payload = {
        "question": "test",
        "model": {"provider": "openai", "name": "gpt", "deployment": "test"},
        "allowed_topics": ["general", "work", "education"],
    }
    response = client.post("/api/validate-topic", json=payload)
    assert response.status_code in (200, 422, 400)
