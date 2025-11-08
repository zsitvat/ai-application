from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.app import create_app
import src.routes.document_routes as document_routes

client = TestClient(create_app())


@pytest.fixture(autouse=True)
def mock_document_service():
    with patch("src.routes.document_routes.DocumentService") as MockService:
        mock_service = MockService.return_value
        mock_service.ingest_documents = AsyncMock()
        mock_service.ingest_positions = AsyncMock()
        yield mock_service


def test_routes_module_exists():
    assert hasattr(document_routes, "router") or hasattr(document_routes, "app")


def test_document_ingest_success(mock_document_service):
    mock_document_service.ingest_documents.return_value = (
        True,
        "ok",
        ["file1.txt"],
        [],
    )
    payload = {
        "files": ["file1.txt"],
        "vector_db_index": "test_index",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embedding_model": {
            "provider": "openai",
            "name": "text-embedding-3-large",
            "type": "embedding",
        },
        "index_schema": [
            {"name": "content", "type": "text"},
            {"name": "source", "type": "text"},
        ],
        "json_data": [],
    }
    response = client.post("/api/documents/ingest", json=payload)
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_document_ingest_missing_index(mock_document_service):
    payload = {
        "files": [],
        "vector_db_index": "",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embedding_model": {},
        "index_schema": {},
        "json_data": None,
    }
    response = client.post("/api/documents/ingest", json=payload)
    assert response.status_code == 400 or response.status_code == 422


def test_document_ingest_error(mock_document_service):
    mock_document_service.ingest_documents.side_effect = Exception("fail")
    payload = {
        "files": ["file1.txt"],
        "vector_db_index": "test_index",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embedding_model": {
            "provider": "openai",
            "name": "text-embedding-3-large",
            "type": "embedding",
        },
        "index_schema": [
            {"name": "content", "type": "text"},
            {"name": "source", "type": "text"},
        ],
        "json_data": [],
    }
    response = client.post("/api/documents/ingest", json=payload)
    assert response.status_code == 500


def test_document_delete_success(mock_document_service):
    mock_document_service.delete_documents.return_value = (True, "deleted", 1)
    response = client.delete("/api/documents/test_index")
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_document_delete_error(mock_document_service):
    mock_document_service.delete_documents.side_effect = Exception("fail")
    response = client.delete("/api/documents/test_index")
    assert response.status_code == 500


def test_ingest_positions_success(mock_document_service):
    mock_document_service.ingest_positions.return_value = (True, "ok", ["f1"], [])
    response = client.post(
        "/api/documents/ingest/positions?file_path=foo.json&index_name=testidx"
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_ingest_positions_error(mock_document_service):
    mock_document_service.ingest_positions.side_effect = Exception("fail")
    response = client.post(
        "/api/documents/ingest/positions?file_path=foo.json&index_name=testidx"
    )
    assert response.status_code == 500
