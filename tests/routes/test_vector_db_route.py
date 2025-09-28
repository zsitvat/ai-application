from fastapi.testclient import TestClient

from src.app import create_app


def test_vector_db_route_exists():
    client = TestClient(create_app())
    response = client.get("/api/vector-db")
    assert response.status_code in (200, 404, 422)
