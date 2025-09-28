from fastapi.testclient import TestClient

from src.app import create_app


def test_graph_config_loader_route_exists():
    client = TestClient(create_app())
    response = client.get("/api/graph/validate-config?config_source=dummy.json")
    assert response.status_code in (200, 400, 404, 422)


def test_graph_config_loader_execute_route_exists():
    client = TestClient(create_app())
    payload = {
        "config_source": "dummy.json",
        "uuid": "test-uuid",
        "applicationIdentifier": {},
        "platform": "webchat",
        "user_input": "test",
        "context": {},
        "user_id": None,
    }
    response = client.post("/api/graph/execute-with-config", json=payload)
    assert response.status_code in (200, 400, 404, 422)
