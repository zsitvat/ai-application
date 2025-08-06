import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from src.schemas.system_schema import (
    HealthCheckResponseSchema,
    LogsResponseSchema,
)
from src.services.system.system_service import SystemService

router = APIRouter(tags=["system"])


def get_system_service():
    return SystemService()


@router.get("/api/health-check", response_model=HealthCheckResponseSchema)
async def health_check(
    system_service: SystemService = Depends(get_system_service),
):
    "Check application health and availability."
    try:
        health_data = await system_service.health_check()

        return HealthCheckResponseSchema(**health_data)

    except Exception as ex:
        logging.getLogger("logger").error(f"Error in health check: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error checking health: {str(ex)}",
        )


@router.get("/api/logs", response_model=LogsResponseSchema)
async def get_logs(
    level: str = Query(None),
    start_date: datetime = Query(None),
    end_date: datetime = Query(None),
    source: str = Query(None),
    limit: int = Query(100),
    page: int = Query(1),
    system_service: SystemService = Depends(get_system_service),
):
    "Retrieve application logs with filtering."
    try:
        logs_data = await system_service.get_logs(
            level=level,
            start_date=start_date,
            end_date=end_date,
            source=source,
            limit=limit,
            page=page,
        )

        return LogsResponseSchema(**logs_data)

    except Exception as ex:
        logging.getLogger("logger").error(f"Error getting logs: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving logs: {str(ex)}",
        )
