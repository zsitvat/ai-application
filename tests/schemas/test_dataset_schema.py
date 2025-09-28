import pytest

from src.schemas.dataset_schema import (
    DatasetCreationError,
    DatasetNotFoundError,
    DatasetRequestSchema,
    DatasetResponseSchema,
    DatasetRunConfigSchema,
    DatasetRunError,
    DatasetRunRequestSchema,
    DatasetUpdateError,
)


def test_dataset_request_schema():
    """Test DatasetRequestSchema instantiation and field values."""
    obj = DatasetRequestSchema(
        dataset_name="test",
        description="desc",
        test_cases=[{"inputs": {"question": "Q"}, "outputs": {"answer": "A"}}],
    )
    assert obj.dataset_name == "test"
    assert obj.description == "desc"
    assert isinstance(obj.test_cases, list)


def test_dataset_run_config_schema():
    """Test DatasetRunConfigSchema instantiation and default values."""
    obj = DatasetRunConfigSchema(
        endpoint="/api/test",
        uuid="1234",
        applicationIdentifier=None,
        platform=None,
        context=None,
        parameters=None,
    )
    assert obj.endpoint == "/api/test"
    assert obj.uuid == "1234"


def test_dataset_run_request_schema():
    """Test DatasetRunRequestSchema instantiation and field values."""
    config = DatasetRunConfigSchema(
        endpoint="/api/test",
        uuid="1234",
        applicationIdentifier=None,
        platform=None,
        context=None,
        parameters=None,
    )
    obj = DatasetRunRequestSchema(config=config)
    assert obj.config.endpoint == "/api/test"


def test_dataset_response_schema():
    """Test DatasetResponseSchema instantiation and field values."""
    obj = DatasetResponseSchema(
        id="id1",
        name="test",
        description="desc",
        test_cases=[{"inputs": {"question": "Q"}, "outputs": {"answer": "A"}}],
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )
    assert obj.id == "id1"
    assert obj.name == "test"
    assert obj.description == "desc"
    assert isinstance(obj.test_cases, list)
    assert obj.created_at == "2024-01-01T00:00:00Z"
    assert obj.updated_at == "2024-01-01T00:00:00Z"


def test_dataset_not_found_error():
    """Test DatasetNotFoundError can be raised and caught."""
    with pytest.raises(DatasetNotFoundError):
        raise DatasetNotFoundError()


def test_dataset_creation_error():
    """Test DatasetCreationError can be raised and caught."""
    with pytest.raises(DatasetCreationError):
        raise DatasetCreationError()


def test_dataset_update_error():
    """Test DatasetUpdateError can be raised and caught."""
    with pytest.raises(DatasetUpdateError):
        raise DatasetUpdateError()


def test_dataset_run_error():
    """Test DatasetRunError can be raised and caught."""
    with pytest.raises(DatasetRunError):
        raise DatasetRunError()
