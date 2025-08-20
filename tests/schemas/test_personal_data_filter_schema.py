from src.schemas.personal_data_filter_schema import (
    PersonalDataFilterRequestSchema,
    PersonalDataFilterResponseSchema,
)


def test_personal_data_filter_request_schema():
    """Test PersonalDataFilterRequestSchema instantiation and field values."""
    obj = PersonalDataFilterRequestSchema(
        text="Sample text", model=None, config=None, enabled=True
    )
    assert obj.text == "Sample text"
    assert obj.model is None
    assert obj.config is None
    assert obj.enabled is True


def test_personal_data_filter_response_schema():
    """Test PersonalDataFilterResponseSchema instantiation and field values."""
    obj = PersonalDataFilterResponseSchema(
        filtered_text="Filtered text", original_text="Original text"
    )
    assert obj.filtered_text == "Filtered text"
    assert obj.original_text == "Original text"
