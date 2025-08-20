import pytest

from src.utils.token_counter import token_counter


@pytest.mark.asyncio
async def test_token_counter_runs():
    result = await token_counter("test", "utf-8")
    assert isinstance(result, int) or result is None


@pytest.mark.asyncio
async def test_token_counter_empty_string():
    result = await token_counter("", "utf-8")
    assert isinstance(result, int) or result is None


@pytest.mark.asyncio
async def test_token_counter_invalid_encoding():
    with pytest.raises(Exception):
        await token_counter("test", "invalid-encoding")
