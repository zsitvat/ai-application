from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.schemas.dataset_schema import (
    DatasetNotFoundError,
    DatasetCreationError,
    DatasetUpdateError,
    DatasetRunError,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_dataset_service():
    with patch("src.routes.dataset_routes.DatasetService") as MockService:
        mock_service = MockService.return_value
        yield mock_service


def test_create_dataset_success(mock_dataset_service):
    mock_dataset_service.create_dataset.return_value = {
        "id": "1",
        "name": "test_ds",
        "description": "desc",
        "test_cases": [],
        "created_at": "2025-08-20T00:00:00Z",
        "updated_at": "2025-08-20T00:00:00Z",
    }
    payload = {"dataset_name": "test_ds", "description": "desc", "test_cases": []}
    response = client.post("/api/dataset", json=payload)
    assert response.status_code == 200
    assert response.json()["name"] == "test_ds"


def test_create_dataset_error(mock_dataset_service):
    mock_dataset_service.create_dataset.side_effect = DatasetCreationError("fail")
    payload = {"dataset_name": "test_ds", "description": "desc", "test_cases": []}
    response = client.post("/api/dataset", json=payload)
    assert response.status_code == 400


def test_create_dataset_unexpected_error(mock_dataset_service):
    mock_dataset_service.create_dataset.side_effect = Exception("fail")
    payload = {"dataset_name": "test_ds", "description": "desc", "test_cases": []}
    response = client.post("/api/dataset", json=payload)
    assert response.status_code == 500


def test_get_dataset_success(mock_dataset_service):
    mock_dataset_service.get_dataset.return_value = {
        "id": "1",
        "name": "test_ds",
        "description": "desc",
        "test_cases": [],
        "created_at": "2025-08-20T00:00:00Z",
        "updated_at": "2025-08-20T00:00:00Z",
    }
    response = client.get("/api/dataset/test_ds")
    assert response.status_code == 200
    assert response.json()["name"] == "test_ds"


def test_get_dataset_not_found(mock_dataset_service):
    mock_dataset_service.get_dataset.side_effect = DatasetNotFoundError("not found")
    response = client.get("/api/dataset/test_ds")
    assert response.status_code == 404


def test_get_dataset_unexpected_error(mock_dataset_service):
    mock_dataset_service.get_dataset.side_effect = Exception("fail")
    response = client.get("/api/dataset/test_ds")
    assert response.status_code == 500


def test_update_dataset_success(mock_dataset_service):
    mock_dataset_service.update_dataset.return_value = {
        "id": "1",
        "name": "test_ds",
        "description": "desc",
        "test_cases": [],
        "created_at": "2025-08-20T00:00:00Z",
        "updated_at": "2025-08-20T00:00:00Z",
    }
    payload = {"dataset_name": "test_ds", "description": "desc", "test_cases": []}
    response = client.patch("/api/dataset/test_ds", json=payload)
    assert response.status_code == 200
    assert response.json()["name"] == "test_ds"


def test_update_dataset_not_found(mock_dataset_service):
    mock_dataset_service.update_dataset.side_effect = DatasetNotFoundError("not found")
    payload = {"dataset_name": "test_ds", "description": "desc", "test_cases": []}
    response = client.patch("/api/dataset/test_ds", json=payload)
    assert response.status_code == 404


def test_update_dataset_error(mock_dataset_service):
    mock_dataset_service.update_dataset.side_effect = DatasetUpdateError("fail")
    payload = {"dataset_name": "test_ds", "description": "desc", "test_cases": []}
    response = client.patch("/api/dataset/test_ds", json=payload)
    assert response.status_code == 400


def test_update_dataset_unexpected_error(mock_dataset_service):
    mock_dataset_service.update_dataset.side_effect = Exception("fail")
    payload = {"dataset_name": "test_ds", "description": "desc", "test_cases": []}
    response = client.patch("/api/dataset/test_ds", json=payload)
    assert response.status_code == 500


def test_run_dataset_success(mock_dataset_service):
    mock_dataset_service.run_dataset.return_value = {"status": "started"}
    payload = {"config": {}}
    response = client.post("/api/dataset/test_ds/run", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "started"


def test_run_dataset_not_found(mock_dataset_service):
    mock_dataset_service.run_dataset.side_effect = DatasetNotFoundError("not found")
    payload = {"config": {}}
    response = client.post("/api/dataset/test_ds/run", json=payload)
    assert response.status_code == 404


def test_run_dataset_error(mock_dataset_service):
    mock_dataset_service.run_dataset.side_effect = DatasetRunError("fail")
    payload = {"config": {}}
    response = client.post("/api/dataset/test_ds/run", json=payload)
    assert response.status_code == 400


def test_run_dataset_unexpected_error(mock_dataset_service):
    mock_dataset_service.run_dataset.side_effect = Exception("fail")
    payload = {"config": {}}
    response = client.post("/api/dataset/test_ds/run", json=payload)
    assert response.status_code == 500


def test_delete_dataset_success(mock_dataset_service):
    mock_dataset_service.delete_dataset.return_value = {
        "dataset_name": "test_ds",
        "message": "deleted",
    }
    response = client.delete("/api/dataset/test_ds")
    assert response.status_code == 200
    assert response.json()["dataset_name"] == "test_ds"


def test_delete_dataset_not_found(mock_dataset_service):
    mock_dataset_service.delete_dataset.side_effect = DatasetNotFoundError("not found")
    response = client.delete("/api/dataset/test_ds")
    assert response.status_code == 404


def test_delete_dataset_unexpected_error(mock_dataset_service):
    mock_dataset_service.delete_dataset.side_effect = Exception("fail")
    response = client.delete("/api/dataset/test_ds")
    assert response.status_code == 500
