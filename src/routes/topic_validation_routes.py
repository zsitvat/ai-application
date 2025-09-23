import logging
import os

from fastapi import APIRouter, Depends, HTTPException

from src.schemas.schema import Model, ModelProviderType, ModelType
from src.schemas.topic_validation_schema import (
    TopicValidationRequestSchema,
    TopicValidationResponseSchema,
)
from src.services.validators.topic_validator.topic_validator_service import (
    TopicValidatorService,
)

router = APIRouter(tags=["topic_validation"])


def get_topic_validator_service():
    return TopicValidatorService()


@router.post("/api/validate-topic", response_model=TopicValidationResponseSchema)
async def validate_topic(
    request: TopicValidationRequestSchema,
    validator_service: TopicValidatorService = Depends(get_topic_validator_service),
):
    "Validate if question belongs to acceptable topics."
    try:
        allowed_topics = getattr(request, "allowed_topics", None)
        invalid_topics = getattr(request, "invalid_topics", None)
        config = getattr(request, "model", None)

        if not config:
            config = Model(
                provider=ModelProviderType.AZURE,
                deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
                type=ModelType.CHAT,
            )

        is_valid, topic, reason = await validator_service.validate_topic(
            question=request.question,
            provider=config.provider.value,
            name=config.name,
            deployment=config.deployment,
            allowed_topics=allowed_topics,
            invalid_topics=invalid_topics,
            raise_on_invalid=False,
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
