from src.schemas.personal_data_filter_schema import (
    PersonalDataFilterRequestSchema,
    PersonalDataFilterResponseSchema,
)


def test_personal_data_filter_request_schema():
    """Test PersonalDataFilterRequestSchema instantiation and field values."""
    obj = PersonalDataFilterRequestSchema(
        text="Sample text",
        model=None,
        sensitive_words=["email", "phone"],
        regex_patterns=[r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
        prompt="test-prompt",
        mask_char="*",
    )
    assert obj.text == "Sample text"
    assert obj.model is None
    assert obj.sensitive_words == ["email", "phone"]
    assert obj.regex_patterns == [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ]
    assert obj.prompt == "test-prompt"
    assert obj.mask_char == "*"


def test_personal_data_filter_response_schema():
    """Test PersonalDataFilterResponseSchema instantiation and field values."""
    obj = PersonalDataFilterResponseSchema(
        filtered_text="Filtered text", original_text="Original text"
    )
    assert obj.filtered_text == "Filtered text"
    assert obj.original_text == "Original text"
