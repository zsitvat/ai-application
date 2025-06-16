import logging

from fastapi import APIRouter, Depends, HTTPException

from schemas.topic_validation_schema import (
    TopicValidationRequestSchema,
    TopicValidationResponseSchema,
)
from services.validators.topic_validator.topic_validator_service import (
    TopicValidatorService,
)

router = APIRouter(tags=["Topic Validation"])


def get_topic_validator_service():
    return TopicValidatorService()


@router.post("/api/validate-topic", response_model=TopicValidationResponseSchema)
async def validate_topic(
    request: TopicValidationRequestSchema,
    validator_service: TopicValidatorService = Depends(get_topic_validator_service),
):
    "Validate if question belongs to acceptable topics."
    try:
        is_valid, topic, reason = await validator_service.validate_topic(
            question=request.question, user_id=request.user_id
        )

        return TopicValidationResponseSchema(
            is_valid=is_valid, topic=topic, reason=reason
        )

    except Exception as ex:
        logging.getLogger("logger").error(f"Error in topic validation: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error validating topic: {str(ex)}",
        )
