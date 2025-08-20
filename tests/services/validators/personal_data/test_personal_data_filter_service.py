from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.schemas.personal_data_filter_schema import PersonalDataFilterConfigSchema
from src.schemas.schema import Model, ModelProviderType, ModelType
from src.services.validators.personal_data.personal_data_filter_service import (
    PersonalDataFilterService,
)


@pytest.fixture
def service():
    return PersonalDataFilterService()


@pytest.mark.asyncio
@patch(
    "src.services.validators.personal_data.personal_data_filter_service.get_chat_model"
)
@patch(
    "src.services.validators.personal_data.personal_data_filter_service.get_prompt_by_type",
    new_callable=AsyncMock,
)
async def test_filter_personal_data_basic(
    mock_get_prompt, mock_get_chat_model, service
):
    mock_get_prompt.return_value = []
    mock_model = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "[FILTERED]"
    mock_model.invoke.return_value = mock_response
    mock_get_chat_model.return_value = mock_model
    data = {"name": "John", "email": "john@example.com"}
    config = PersonalDataFilterConfigSchema(
        model=Model(
            provider=ModelProviderType.OPENAI,
            name="gpt-3",
            type=ModelType.CHAT,
            deployment=None,
        ),
        prompt=None,
        sensitive_words=None,
        regex_patterns=None,
    )
    result = await service.filter_personal_data(data, config)
    assert isinstance(result, tuple)
    assert result[0] == "[FILTERED]"
    assert result[1] == data
    mock_get_prompt.return_value = []


@pytest.mark.asyncio
@patch(
    "src.services.validators.personal_data.personal_data_filter_service.get_chat_model"
)
@patch(
    "src.services.validators.personal_data.personal_data_filter_service.get_prompt_by_type",
    new_callable=AsyncMock,
)
async def test_filter_personal_data_empty(
    mock_get_prompt, mock_get_chat_model, service
):
    mock_get_prompt.return_value = []
    mock_model = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = ""
    mock_model.invoke.return_value = mock_response
    mock_get_chat_model.return_value = mock_model
    config = PersonalDataFilterConfigSchema(
        model=Model(
            provider=ModelProviderType.OPENAI,
            name="gpt-3",
            type=ModelType.CHAT,
            deployment=None,
        ),
        prompt=None,
        sensitive_words=None,
        regex_patterns=None,
    )
    result = await service.filter_personal_data({}, config)
    assert isinstance(result, tuple)
    assert result[0] == ""
    assert result[1] == {}


@pytest.mark.asyncio
@patch(
    "src.services.validators.personal_data.personal_data_filter_service.get_chat_model"
)
@patch(
    "src.services.validators.personal_data.personal_data_filter_service.get_prompt_by_type",
    new_callable=AsyncMock,
)
async def test_filter_personal_data_none(mock_get_prompt, mock_get_chat_model, service):
    mock_get_prompt.return_value = []
    mock_model = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = ""
    mock_model.invoke.return_value = mock_response
    mock_get_chat_model.return_value = mock_model
    config = PersonalDataFilterConfigSchema(
        model=Model(
            provider=ModelProviderType.OPENAI,
            name="gpt-3",
            type=ModelType.CHAT,
            deployment=None,
        ),
        prompt=None,
        sensitive_words=None,
        regex_patterns=None,
    )
    result = await service.filter_personal_data(None, config)
    assert isinstance(result, tuple)
    assert result[0] == ""
    assert result[1] is None


@pytest.mark.asyncio
@patch(
    "src.services.validators.personal_data.personal_data_filter_service.get_chat_model"
)
@patch(
    "src.services.validators.personal_data.personal_data_filter_service.get_prompt_by_type",
    new_callable=AsyncMock,
)
async def test_filter_personal_data_edge(mock_get_prompt, mock_get_chat_model, service):
    mock_get_prompt.return_value = []
    mock_model = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = ""
    mock_model.invoke.return_value = mock_response
    mock_get_chat_model.return_value = mock_model
    config = PersonalDataFilterConfigSchema(
        model=Model(
            provider=ModelProviderType.OPENAI,
            name="gpt-3",
            type=ModelType.CHAT,
            deployment=None,
        ),
        prompt=None,
        sensitive_words=None,
        regex_patterns=None,
    )
    result = await service.filter_personal_data(["notadict"], config)
    assert isinstance(result, tuple)
    assert result[0] == ""
    assert result[1] == ["notadict"]
