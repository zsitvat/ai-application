from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.app import create_app
from src.routes.personal_data_filter_routes import get_personal_data_filter_service


@pytest.fixture(autouse=True)
def mock_personal_data_filter_service(monkeypatch):
    mock_service = MagicMock()

    async def always_return(*args, **kwargs):
        return ("filtered text", "original text")

    mock_service.filter_personal_data = AsyncMock(side_effect=always_return)
    monkeypatch.setattr(
        "src.routes.personal_data_filter_routes.get_personal_data_filter_service",
        lambda: mock_service,
    )
    return mock_service


def test_personal_data_filter_route_exists(mock_personal_data_filter_service):
    app = create_app()
    app.dependency_overrides[get_personal_data_filter_service] = (
        lambda: mock_personal_data_filter_service
    )
    client = TestClient(app)
    payload = {
        "text": "Sensitive info here.",
        "config": {
            "sensitive_words": ["info"],
            "regex_patterns": ["\\binfo\\b"],
            "model": {
                "provider": "openai",
                "deployment": "test",
                "name": "gpt-4o-mini",
                "type": "chat",
            },
            "prompt": "Mask sensitive info",
        },
        "enabled": True,
    }
    response = client.post("/api/personal-data-filter", json=payload)
    assert response.status_code in (200, 422, 400)
    if response.status_code == 200:
        data = response.json()
        assert "filtered_text" in data
        assert "original_text" in data
