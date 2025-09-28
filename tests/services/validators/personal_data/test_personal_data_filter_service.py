from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.schemas.schema import Model, ModelProviderType, ModelType
from src.services.validators.personal_data.personal_data_filter_service import (
    PersonalDataFilterService,
)


@pytest.fixture
def service():
    return PersonalDataFilterService()


@pytest.fixture
def test_model():
    return Model(
        provider=ModelProviderType.AZURE,
        name="gpt-4o-mini",
        type=ModelType.CHAT,
        deployment="test-deployment",
    )


@pytest.mark.asyncio
async def test_apply_regex_replacements(service):
    text = "Email: john@example.com, Phone: +36-30-123-4567"
    regex_patterns = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        r"\+?36[- ]?\d{1,2}[- ]?\d{3}[- ]?\d{4}",
    ]

    result = service.apply_regex_replacements(text, regex_patterns, "X")

    assert "XXXXXXXXXXXXXXXX" in result
    assert "XXXXXXXXXXXXXXX" in result
    assert "Email:" in result
    assert "Phone:" in result


@pytest.mark.asyncio
async def test_apply_regex_replacements_invalid_pattern(service):
    text = "Test text with email@test.com"
    regex_patterns = ["[invalid"]

    result = service.apply_regex_replacements(text, regex_patterns, "*")

    assert result == text


@pytest.mark.asyncio
@patch("src.services.validators.personal_data.personal_data_filter_service.get_model")
@patch(
    "src.services.validators.personal_data.personal_data_filter_service.get_prompt_by_type",
    new_callable=AsyncMock,
)
async def test_filter_personal_data_with_ai(
    mock_get_prompt, mock_get_model, service, test_model
):
    mock_prompt = MagicMock()
    mock_prompt.format_messages.return_value = ["test message"]
    mock_get_prompt.return_value = mock_prompt

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Filtered content"
    mock_llm.invoke.return_value = mock_response
    mock_get_model.return_value = mock_llm

    result = await service.filter_personal_data(
        text="John Doe email: john@example.com",
        model=test_model,
        sensitive_words=["email"],
        regex_patterns=[r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
        prompt="personal-data-filter-prompt",
        mask_char="*",
    )

    assert result == "Filtered content"
    mock_get_prompt.assert_called_once_with("personal-data-filter-prompt")
    mock_get_model.assert_called_once()


@pytest.mark.asyncio
async def test_filter_personal_data_regex_only(service, test_model):
    text = "Contact: john@example.com, Phone: +36-30-123-4567"
    regex_patterns = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        r"\+?36[- ]?\d{1,2}[- ]?\d{3}[- ]?\d{4}",
    ]

    result = await service.filter_personal_data(
        text=text, model=test_model, regex_patterns=regex_patterns, mask_char="X"
    )

    assert "XXXXXXXXXXXXXXXX" in result
    assert "XXXXXXXXXXXXXXX" in result
    assert "Contact:" in result


@pytest.mark.asyncio
async def test_filter_personal_data_empty_text(service, test_model):
    result = await service.filter_personal_data(
        text="", model=test_model, prompt="test-prompt"
    )

    assert result == ""


@pytest.mark.asyncio
async def test_filter_personal_data_none_text(service, test_model):
    result = await service.filter_personal_data(
        text=None, model=test_model, prompt="test-prompt"
    )

    assert result is None


@pytest.mark.asyncio
async def test_filter_personal_data_whitespace_only(service, test_model):
    result = await service.filter_personal_data(
        text="   \n\t  ", model=test_model, prompt="test-prompt"
    )

    assert result == "   \n\t  "


@pytest.mark.asyncio
@patch("src.services.validators.personal_data.personal_data_filter_service.get_model")
async def test_filter_personal_data_ai_error(mock_get_model, service, test_model):
    mock_get_model.side_effect = Exception("AI model error")

    with pytest.raises(Exception, match="AI model error"):
        await service.filter_personal_data(
            text="Test text", model=test_model, prompt="test-prompt"
        )
