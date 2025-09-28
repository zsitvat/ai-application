import pytest

from src.utils.get_search_scores import create_score_string, get_scores_as_string


@pytest.mark.asyncio
async def test_create_score_string_runs():
    result = await create_score_string({}, 0)
    assert isinstance(result, str) or result is None


@pytest.mark.asyncio
async def test_create_score_string_empty():
    result = await create_score_string({}, None)
    assert isinstance(result, str) or result is None


@pytest.mark.asyncio
async def test_create_score_string_invalid():
    result = await create_score_string(None, None)
    assert result == "Invalid document data"


@pytest.mark.asyncio
async def test_get_scores_as_string_runs():
    result = await get_scores_as_string([{}])
    assert isinstance(result, str) or result is None


@pytest.mark.asyncio
async def test_get_scores_as_string_empty():
    result = await get_scores_as_string([])
    assert isinstance(result, str) or result is None


@pytest.mark.asyncio
async def test_get_scores_as_string_invalid():
    result = await get_scores_as_string(None)
    assert result == "No documents found."
