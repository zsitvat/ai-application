from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.dataset.dataset_service import (
    DatasetCreationError,
    DatasetNotFoundError,
    DatasetRunError,
    DatasetService,
    DatasetUpdateError,
)


@pytest.fixture
def service():
    svc = DatasetService()
    svc.client = MagicMock()
    return svc


def test_create_dataset_success(service):
    mock_ds = MagicMock()
    mock_ds.id = "1"
    mock_ds.name = "ds"
    mock_ds.description = "desc"
    mock_ds.created_at = None
    mock_ds.url = "url"
    mock_ds.tags = []
    mock_ds.extra = {}
    service.client.create_dataset.return_value = mock_ds
    service._update_test_cases = MagicMock()
    result = service.create_dataset("ds", "desc", [{"inputs": {}, "outputs": {}}])
    assert result["id"] == "1"
    assert result["name"] == "ds"


def test_create_dataset_no_test_cases(service):
    service.client.create_dataset.return_value = MagicMock(
        id="1",
        name="ds",
        description="desc",
        created_at=None,
        url="url",
        tags=[],
        extra={},
    )
    result = service.create_dataset("ds", "desc", None)
    assert result["test_cases_count"] == 0


def test_create_dataset_error(service):
    service.client.create_dataset.side_effect = Exception("fail")
    with pytest.raises(DatasetCreationError):
        service.create_dataset("ds", "desc", [])


def test_get_dataset_success(service):
    ds = MagicMock(
        id="1",
        name="ds",
        description="desc",
        created_at=None,
        modified_at=None,
        url="url",
        tags=[],
        extra={},
    )
    service.client.list_datasets.return_value = [ds]
    ex = MagicMock(id="2", inputs={}, outputs={}, metadata={}, created_at=None)
    service.client.list_examples.return_value = [ex]
    result = service.get_dataset("ds")
    assert result["id"] == "1"
    assert result["test_cases_count"] == 1


def test_get_dataset_not_found(service):
    service.client.list_datasets.return_value = []
    with pytest.raises(DatasetNotFoundError):
        service.get_dataset("ds")


def test_get_dataset_error(service):
    service.client.list_datasets.side_effect = Exception("fail")
    with pytest.raises(DatasetNotFoundError):
        service.get_dataset("ds")


def test_update_dataset_success(service):
    ds = MagicMock(
        id="1",
        name="ds",
        description="desc",
        created_at=None,
        modified_at=None,
        url="url",
        tags=[],
        extra={},
    )
    service.client.list_datasets.return_value = [ds]
    ds.update = MagicMock()
    service._update_test_cases = MagicMock()
    service.get_dataset = MagicMock(return_value={"id": "1"})
    result = service.update_dataset(
        "ds", name="new", description="desc", test_cases=[{"inputs": {}, "outputs": {}}]
    )
    assert result["id"] == "1"


def test_update_dataset_not_found(service):
    service.client.list_datasets.return_value = []
    with pytest.raises(DatasetNotFoundError):
        service.update_dataset("ds", name="new")


def test_update_dataset_error(service):
    service.client.list_datasets.side_effect = Exception("fail")
    with pytest.raises(DatasetUpdateError):
        service.update_dataset("ds", name="new")


def test_run_dataset_success(service):
    with patch("asyncio.create_task", MagicMock()):
        result = service.run_dataset("ds", config={})
        assert result["status"] == "started"


def test_run_dataset_error(service):
    with patch("asyncio.create_task", side_effect=Exception("fail")):
        with pytest.raises(DatasetRunError):
            service.run_dataset("ds", config={})


def test_delete_dataset_success(service):
    ds = MagicMock(id="1", name="ds")
    service.client.list_datasets.return_value = [ds]
    service.client.delete_dataset = MagicMock()
    result = service.delete_dataset("ds")
    assert result["dataset_name"] == "ds"


def test_delete_dataset_not_found(service):
    service.client.list_datasets.return_value = []
    with pytest.raises(DatasetNotFoundError):
        service.delete_dataset("ds")


def test_delete_dataset_error(service):
    service.client.list_datasets.side_effect = Exception("fail")
    with pytest.raises(DatasetUpdateError):
        service.delete_dataset("ds")


def test_prepare_examples_from_test_cases(service):
    cases = [{"inputs": {"a": 1}, "outputs": {"b": 2}, "metadata": {"c": 3}}]
    result = service._prepare_examples_from_test_cases(cases)
    assert result[0]["inputs"] == {"a": 1}
    assert result[0]["outputs"] == {"b": 2}
    assert result[0]["metadata"] == {"c": 3}


def test_update_test_cases(service):
    service.client.create_example = MagicMock()
    cases = [{"inputs": {"a": 1}, "outputs": {"b": 2}, "metadata": {"c": 3}}]
    service._update_test_cases("id", cases)
    service.client.create_example.assert_called()


@pytest.mark.asyncio
async def test_call_graph_api_success(service):
    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = MagicMock()
        mock_instance.post = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value={"result": "ok"}),
                raise_for_status=MagicMock(),
            )
        )
        mock_client.return_value.__aenter__.return_value = mock_instance
        result = await service._call_graph_api("question", config={})
        assert "result" in result


@pytest.mark.asyncio
async def test_call_graph_api_error(service):
    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = MagicMock()
        mock_instance.post = AsyncMock(side_effect=Exception("fail"))
        mock_client.return_value.__aenter__.return_value = mock_instance
        with pytest.raises(Exception):
            await service._call_graph_api("question", config={})


def test_create_result_dict_success(service):
    result = service._create_result_dict("q", {"a": 1}, None, {})
    assert result["success"] is True
    assert result["response"] == {"a": 1}


def test_create_result_dict_error(service):
    result = service._create_result_dict("q", {}, Exception("fail"), {})
    assert result["success"] is False
    assert result["error"] == "fail"


def test_create_llm_function_success(service):
    func = service._create_llm_function({})
    with patch.object(service, "_call_graph_api", return_value={"result": "ok"}):
        result = func({"user_input": "question"})
        assert "success" in result


def test_create_llm_function_no_question(service):
    func = service._create_llm_function({})
    result = func({})
    assert "error" in result
