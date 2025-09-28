import pytest

from src.schemas.token_validation_schema import (
    TokenCountResult,
    TokenEstimationResult,
    TruncationResult,
)


def test_token_count_result():
    """Test TokenCountResult instantiation and field values."""
    obj = TokenCountResult(
        text="Hello world",
        token_count=2,
        character_count=11,
        encoding_used="utf-8",
        chars_per_token=5.5,
    )
    assert obj.text == "Hello world"
    assert obj.token_count == 2
    assert obj.character_count == 11
    assert obj.encoding_used == "utf-8"
    assert obj.chars_per_token == pytest.approx(5.5)


def test_truncation_result():
    """Test TruncationResult instantiation and field values."""
    obj = TruncationResult(
        original_text="Hello world",
        truncated_text="Hello",
        original_tokens=2,
        final_tokens=1,
        was_truncated=True,
        encoding_used="utf-8",
    )
    assert obj.original_text == "Hello world"
    assert obj.truncated_text == "Hello"
    assert obj.original_tokens == 2
    assert obj.final_tokens == 1
    assert obj.was_truncated is True
    assert obj.encoding_used == "utf-8"


def test_token_estimation_result():
    """Test TokenEstimationResult instantiation and field values."""
    obj = TokenEstimationResult(
        estimated_characters=100,
        target_tokens=20,
        encoding_used="utf-8",
        chars_per_token_ratio=5.0,
    )
    assert obj.estimated_characters == 100
    assert obj.target_tokens == 20
    assert obj.encoding_used == "utf-8"
    assert obj.chars_per_token_ratio == pytest.approx(5.0)
