import logging

from fastapi import APIRouter, Depends, HTTPException

from src.schemas.personal_data_filter_schema import (
    PersonalDataFilterRequestSchema,
    PersonalDataFilterResponseSchema,
)
from src.services.validators.personal_data.personal_data_filter_service import (
    PersonalDataFilterService,
)

router = APIRouter(tags=["personal_data_sfilter"])


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
    """Filter personal and sensitive data from text."""

    try:
        filtered_text, original_text = await filter_service.filter_personal_data(
            text=request.text,
            config=request.config,
        )

        return PersonalDataFilterResponseSchema(
            filtered_text=filtered_text, original_text=original_text
        )

    except Exception as ex:
        logging.getLogger("logger").error(
            f"Error in personal data filtering: {str(ex)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error filtering personal data: {str(ex)}",
        )
