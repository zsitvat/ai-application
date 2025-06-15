from fastapi import APIRouter, HTTPException, Depends
import logging

from schemas.dataset_schema import (
    DatasetCreateRequestSchema,
    DatasetUpdateRequestSchema,
    DatasetResponseSchema,
    DatasetRunRequestSchema,
)
from services.dataset.dataset_service import DatasetService

router = APIRouter(tags=["Dataset"])


def get_dataset_service():
    return DatasetService()


@router.post("/api/dataset", response_model=DatasetResponseSchema)
async def create_dataset(
    request: DatasetCreateRequestSchema,
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    "Create a new test dataset."
    try:
        dataset = await dataset_service.create_dataset(
            name=request.dataset_names,
            description=request.description,
            test_cases=request.test_cases,
        )

        return DatasetResponseSchema(**dataset)

    except Exception as ex:
        logging.getLogger("logger").error(f"Error creating dataset: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating dataset: {str(ex)}",
        )


@router.get("/api/dataset/{dataset_name}", response_model=DatasetResponseSchema)
async def get_dataset(
    dataset_name: str,
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    "Get test dataset by name."
    try:
        dataset = await dataset_service.get_dataset(dataset_name=dataset_name)

        return DatasetResponseSchema(**dataset)

    except Exception as ex:
        logging.getLogger("logger").error(f"Error getting dataset: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting dataset: {str(ex)}",
        )


@router.patch("/api/dataset/{dataset_name}", response_model=DatasetResponseSchema)
async def update_dataset(
    dataset_name: str,
    request: DatasetUpdateRequestSchema,
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    "Update test dataset."
    try:
        dataset = await dataset_service.update_dataset(
            dataset_name=dataset_name,
            description=request.description,
            test_cases=request.test_cases,
        )

        return DatasetResponseSchema(**dataset)

    except Exception as ex:
        logging.getLogger("logger").error(f"Error updating dataset: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating dataset: {str(ex)}",
        )


@router.post("/api/dataset/{dataset_name}/run", response_model=DatasetResponseSchema)
async def run_dataset(
    dataset_name: str,
    request: DatasetRunRequestSchema,
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    "Run test dataset against multi-agent graph."
    try:
        results = await dataset_service.run_dataset(
            dataset_name=dataset_name, config=request.config
        )

        return DatasetResponseSchema(**results)

    except Exception as ex:
        logging.getLogger("logger").error(f"Error running dataset: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error running dataset: {str(ex)}",
        )
