from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.app import create_app


@pytest.fixture(autouse=True)
def mock_web_scraping_service(monkeypatch):
    mock_service = MagicMock()
    mock_service.scrape_websites = AsyncMock(
        return_value=(
            True,
            "Scraping successful",
            ["http://example.com"],
            [],
            {"http://example.com": "content"},
        )
    )
    monkeypatch.setattr(
        "src.routes.web_scraping_routes.get_web_scraping_service",
        lambda: mock_service,
    )
    return mock_service


def test_web_scraping_route_error(monkeypatch):
    app = create_app()
    client = TestClient(app)

    # Patch the service to raise an exception
    async def raise_exc(*args, **kwargs):
        raise Exception("Service error")

    mock_service = MagicMock()
    mock_service.scrape_websites = AsyncMock(side_effect=raise_exc)
    monkeypatch.setattr(
        "src.routes.web_scraping_routes.get_web_scraping_service",
        lambda: mock_service,
    )
    payload = {
        "urls": ["http://example.com"],
        "max_depth": 1,
        "output_type": "txt",
        "output_path": None,
        "vector_db_index": None,
        "allowed_domains": ["example.com"],
        "content_selectors": ["body"],
        "excluded_selectors": [],
        "embedding_model": {
            "provider": "openai",
            "name": "text-embedding-3-large",
            "type": "embedding",
        },
    }
    response = client.post("/api/web-scraping", json=payload)
    assert response.status_code == 500
    assert "Error scraping websites" in response.json()["detail"]
