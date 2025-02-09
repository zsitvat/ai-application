from fastapi.testclient import TestClient
from src.app import create_app


async def test_agent_route_responds():
    """
    Test that the agent route responds with a valid status.
    """

    client = await TestClient(create_app())

    json = {
        "prompt": "question",
        "user_input": "Milyen az időjárás Győrben?",
        "model": {
            "name": "gpt-4o-mini",
            "type": "chat",
            "provider": "openai",
            "temperature": 0,
        },
        "tools": [
            {
                "name": "retriver_tool",
                "vector_db_path": "deeplake_databases/deeplake_db_pdf",
                "model": {
                    "name": "text-embedding-3-large",
                    "provider": "openai",
                },
                "search_kwargs": {"search_type": "similarity"},
            },
            {"name": "web_search_tool", "engine": "google"},
        ],
    }

    response = await client.post("api/agent/answer", json=json)
    assert response.status_code == 200
    assert "answer" in response.json()
