from fastapi.testclient import TestClient


from src.app import create_app


async def test_create_app():
    """
    Test that the FastAPI application initializes correctly.
    """
    app = create_app()
    client = TestClient(app)
    response = await client.get("/")
    assert response.status_code in [200, 404]


async def test_routers():
    """
    Test that the routers are included in the app.
    """
    app = create_app()

    assert any(route.path == "/api/agent/answer" for route in app.routes)
    assert any(route.path == "/api/vector_db/create" for route in app.routes)
