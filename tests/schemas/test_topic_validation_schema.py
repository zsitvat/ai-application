from src.schemas.topic_validation_schema import (
    TopicValidationRequestSchema,
    TopicValidationResponseSchema,
)


def test_topic_validation_request_schema():
    """Test TopicValidationRequestSchema instantiation and field values."""
    obj = TopicValidationRequestSchema(
        question="Is AI used in recruitment?",
        model=None,
        allowed_topics=["AI", "Recruitment"],
        invalid_topics=None,
        enabled=True,
    )
    assert obj.question == "Is AI used in recruitment?"
    assert obj.model is None
    assert obj.allowed_topics == ["AI", "Recruitment"]
    assert obj.invalid_topics is None
    assert obj.enabled is True


def test_topic_validation_response_schema():
    """Test TopicValidationResponseSchema instantiation and field values."""
    obj = TopicValidationResponseSchema(
        is_valid=True, topic="AI", reason="Topic is allowed"
    )
    assert obj.is_valid is True
    assert obj.topic == "AI"
    assert obj.reason == "Topic is allowed"
