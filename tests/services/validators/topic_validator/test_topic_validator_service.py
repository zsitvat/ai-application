from unittest.mock import AsyncMock, patch

import pytest

from src.services.validators.topic_validator.topic_validator_service import (
    TopicValidatorService,
)


@pytest.fixture
def service():
    return TopicValidatorService()


@pytest.mark.asyncio
async def test_validate_topic_basic(service):
    topic = "AI"
    provider = "openai"
    name = "gpt-4o-mini"
    deployment = ""
    allowed_topics = ["AI", "HR", "Finance"]
    with patch.object(service, "_classify_with_llm", new=AsyncMock(return_value="AI")):
        result = await service.validate_topic(
            topic, provider, name, deployment, allowed_topics=allowed_topics
        )
        assert isinstance(result, tuple)
        assert result[0] is True
        assert result[1] == "AI"


@pytest.mark.asyncio
async def test_validate_topic_empty(service):
    provider = "openai"
    name = "gpt-4o-mini"
    deployment = "default"
    allowed_topics = ["AI", "HR", "Finance"]
    with patch.object(service, "_classify_with_llm", new=AsyncMock(return_value="AI")):
        result = await service.validate_topic(
            "", provider, name, deployment, allowed_topics=allowed_topics
        )
        assert isinstance(result, tuple)
        assert result[0] is False


@pytest.mark.asyncio
async def test_validate_topic_edge(service):
    provider = "openai"
    name = "gpt-4o-mini"
    deployment = "default"
    allowed_topics = ["AI", "HR", "Finance"]
    with patch.object(service, "_classify_with_llm", new=AsyncMock(return_value="AI")):
        with pytest.raises(TypeError):
            await service.validate_topic(
                123, provider, name, deployment, allowed_topics=allowed_topics
            )
