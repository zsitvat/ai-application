from fastapi.testclient import TestClient

from src.app import create_app


def test_create_app():
    """
    Test that the FastAPI application initializes correctly.
    """
    app = create_app()
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code in [200, 404]
