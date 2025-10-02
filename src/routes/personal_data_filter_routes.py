import os

from fastapi import APIRouter, Depends, HTTPException

from src.schemas.personal_data_filter_schema import (
    PersonalDataFilterRequestSchema,
    PersonalDataFilterResponseSchema,
)
from schemas.model_schema import Model, ModelProviderType, ModelType
from src.services.logger.logger_service import LoggerService
from src.services.validators.personal_data.personal_data_filter_service import (
    PersonalDataFilterService,
)

logger = LoggerService().setup_logger()

router = APIRouter(tags=["personal_data_filter"])


def get_personal_data_filter_service():
    return PersonalDataFilterService()


@router.post(
    "/api/personal-data-filter", response_model=PersonalDataFilterResponseSchema
)
async def filter_personal_data(
    request: PersonalDataFilterRequestSchema,
    filter_service: PersonalDataFilterService = Depends(
        get_personal_data_filter_service
    ),
):

    try:
        if request.model is None:
            request.model = Model(
                provider=ModelProviderType.AZURE,
                name="gpt-4o-mini",
                deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
                type=ModelType.CHAT,
            )

        if request.sensitive_words is None:
            request.sensitive_words = ["phone", "email", "name", "address"]

        if request.regex_patterns is None:
            request.regex_patterns = [
                r"\+36\s?\d{2}\s?\d{3}\s?\d{4}",
                r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            ]

        filtered_text = await filter_service.filter_personal_data(
            text=request.text,
            model=request.model,
            sensitive_words=request.sensitive_words,
            regex_patterns=request.regex_patterns,
            prompt=request.prompt,
            mask_char=request.mask_char,
        )

        return PersonalDataFilterResponseSchema(
            filtered_text=filtered_text, original_text=request.text
        )

    except Exception as ex:
        logger.error(
            f"[PersonalDataFilterRoutes] Error in personal data filtering: {str(ex)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error filtering personal data: {str(ex)}",
        )
