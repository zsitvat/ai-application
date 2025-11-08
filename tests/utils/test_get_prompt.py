import pytest
from unittest.mock import patch

from src.utils.get_prompt import _is_integer, get_prompt_by_type


@pytest.mark.asyncio
async def test_get_prompt_by_type_basic():

    with patch("langchain.hub.pull", return_value=["prompt"]):
        result = await get_prompt_by_type(
            "default", tracer_type="langsmith", cache_ttl=60
        )
        assert isinstance(result, list) or result is not None


@pytest.mark.asyncio
async def test_get_prompt_by_type_invalid():

    with patch("langchain.hub.pull", side_effect=Exception("fail")):
        with pytest.raises(Exception):
            await get_prompt_by_type("invalid", tracer_type="langsmith", cache_ttl=60)


@pytest.mark.asyncio
async def test_get_prompt_by_type_other_tracer():

    with patch("langchain.hub.pull", return_value=["prompt"]):
        result = await get_prompt_by_type("default", tracer_type="other", cache_ttl=60)
        assert isinstance(result, list) or result is not None


@pytest.mark.asyncio
async def test_get_prompt_by_type_other_tracer_exception():

    with patch("langchain.hub.pull", side_effect=Exception("fail")):
        with pytest.raises(Exception):
            await get_prompt_by_type("default", tracer_type="other", cache_ttl=60)


def test_is_integer_true():
    assert _is_integer("123") is True
    assert _is_integer(123) is True


def test_is_integer_false():
    assert _is_integer("abc") is False
    assert _is_integer(None) is False
