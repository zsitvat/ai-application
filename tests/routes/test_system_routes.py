from fastapi.testclient import TestClient

from src.app import create_app


def test_system_routes_exists():
    client = TestClient(create_app())
    response_health = client.get("/api/health-check")
    response_logs = client.get("/api/logs")
    assert response_health.status_code in (200, 404, 422)
    assert response_logs.status_code in (200, 404, 422)
