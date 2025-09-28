from fastapi import APIRouter, Depends, HTTPException

from src.schemas.dataset_schema import (
    DatasetCreationError,
    DatasetNotFoundError,
    DatasetRequestSchema,
    DatasetResponseSchema,
    DatasetRunError,
    DatasetRunRequestSchema,
    DatasetUpdateError,
)
from src.services.dataset.dataset_service import DatasetService
from src.services.logger.logger_service import LoggerService

logger = LoggerService().setup_logger()

router = APIRouter(tags=["dataset"])


def get_dataset_service():
    return DatasetService()


@router.post("/api/dataset", response_model=DatasetResponseSchema)
def create_dataset(
    request: DatasetRequestSchema,
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    "Create a new test dataset."
    try:
        dataset = dataset_service.create_dataset(
            name=request.dataset_name,
            description=request.description,
            test_cases=request.test_cases or [],
        )

        return DatasetResponseSchema(**dataset)

    except DatasetCreationError as ex:
        logger.error(f"[DatasetRoutes] Error creating dataset: {str(ex)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error creating dataset: {str(ex)}",
        )
    except Exception as ex:
        logger.error(f"[DatasetRoutes] Unexpected error creating dataset: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error creating dataset: {str(ex)}",
        )


@router.get("/api/dataset/{dataset_name}", response_model=DatasetResponseSchema)
def get_dataset(
    dataset_name: str,
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    "Get test dataset by name."
    try:
        dataset = dataset_service.get_dataset(dataset_name=dataset_name)

        return DatasetResponseSchema(**dataset)

    except DatasetNotFoundError as ex:
        logger.error(f"[DatasetRoutes] Dataset not found: {str(ex)}")
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {str(ex)}",
        )
    except Exception as ex:
        logger.error(f"[DatasetRoutes] Unexpected error getting dataset: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error getting dataset: {str(ex)}",
        )


@router.patch("/api/dataset/{dataset_name}", response_model=DatasetResponseSchema)
def update_dataset(
    request: DatasetRequestSchema,
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    "Update test dataset."
    try:
        dataset = dataset_service.update_dataset(
            dataset_name=request.dataset_name,
            description=request.description,
            test_cases=request.test_cases,
        )

        return DatasetResponseSchema(**dataset)

    except DatasetNotFoundError as ex:
        logger.error(f"[DatasetRoutes] Dataset not found: {str(ex)}")
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {str(ex)}",
        )
    except DatasetUpdateError as ex:
        logger.error(f"[DatasetRoutes] Error updating dataset: {str(ex)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error updating dataset: {str(ex)}",
        )
    except Exception as ex:
        logger.error(f"[DatasetRoutes] Unexpected error updating dataset: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error updating dataset: {str(ex)}",
        )


@router.post("/api/dataset/{dataset_name}/run")
async def run_dataset(
    dataset_name: str,
    request: DatasetRunRequestSchema,
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    "Run test dataset against multi-agent graph."
    try:
        results = dataset_service.run_dataset(
            dataset_name=dataset_name, config=request.config
        )

        return results

    except DatasetNotFoundError as ex:
        logger.error(f"[DatasetRoutes] Dataset not found: {str(ex)}")
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {str(ex)}",
        )
    except DatasetRunError as ex:
        logger.error(f"[DatasetRoutes] Error running dataset: {str(ex)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error running dataset: {str(ex)}",
        )
    except Exception as ex:
        logger.error(f"[DatasetRoutes] Unexpected error running dataset: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error running dataset: {str(ex)}",
        )


@router.delete("/api/dataset/{dataset_name}")
def delete_dataset(
    dataset_name: str,
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    "Delete a test dataset."
    try:
        result = dataset_service.delete_dataset(dataset_name=dataset_name)
        return result

    except DatasetNotFoundError as ex:
        logger.error(f"[DatasetRoutes] Dataset not found: {str(ex)}")
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {str(ex)}",
        )
    except Exception as ex:
        logger.error(f"[DatasetRoutes] Unexpected error deleting dataset: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error deleting dataset: {str(ex)}",
        )
