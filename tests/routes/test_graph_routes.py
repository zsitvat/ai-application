from fastapi.testclient import TestClient

from src.app import create_app


def test_graph_route_exists():
    client = TestClient(create_app())
    response = client.post("/api/graph", json={})
    assert response.status_code in (200, 422)


def test_graph_stream_route_exists():
    client = TestClient(create_app())
    response = client.post("/api/graph/stream", json={})
    assert response.status_code in (200, 422)
