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
    class FailingService:
        async def scrape_websites(self, *args, **kwargs):
            raise RuntimeError("Service error")

    monkeypatch.setattr(
        "src.routes.web_scraping_routes.get_web_scraping_service",
        lambda: FailingService(),
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
    try:
        error_detail = response.json().get("detail", "")
    except Exception:
        error_detail = response.text
    assert "Unexpected error scraping websites" in error_detail
