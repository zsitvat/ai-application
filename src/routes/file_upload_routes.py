from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.schemas.file_upload_schema import (
    Base64FileUpload,
    FileUploadResponse,
    StorageBackendConfig,
)
from services.document.file_upload_service import FileUploadService
from src.services.logger.logger_service import LoggerService

router = APIRouter(prefix="/api/files", tags=["file_upload"])
logger = LoggerService().setup_logger()

# Initialize file upload service with default configuration
file_upload_service = FileUploadService(StorageBackendConfig())


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...), thread_id: Optional[str] = Form(None)
) -> FileUploadResponse:
    """Upload a file using configured storage backend."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Read file content
    file_content = await file.read()
    mime_type = file.content_type or "application/octet-stream"

    try:
        return await file_upload_service.upload_file(
            file_content, file.filename, mime_type, thread_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.post("/upload-base64", response_model=FileUploadResponse)
async def upload_base64_file(request: Base64FileUpload) -> FileUploadResponse:
    """Upload a base64 encoded file using configured storage backend."""
    try:
        return await file_upload_service.upload_base64_file(
            request.filename, request.file_data, request.thread_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 file upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint for file upload service."""
    return {
        "status": "healthy",
        "service": "file_upload",
        "supported_backends": ["local", "s3", "drive"],
    }
