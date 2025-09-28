from fastapi import APIRouter, Depends, HTTPException

from src.schemas.document_schema import (
    DocumentDeleteResponseSchema,
    DocumentIngestRequestSchema,
    DocumentIngestResponseSchema,
)
from src.services.document.document_service import DocumentService
from src.services.logger.logger_service import LoggerService

logger = LoggerService().setup_logger()

router = APIRouter(tags=["documents"])


def get_document_service():
    return DocumentService()


@router.post("/api/documents/ingest", response_model=DocumentIngestResponseSchema)
async def ingest_documents(
    request: DocumentIngestRequestSchema,
    document_service: DocumentService = Depends(get_document_service),
):
    "Process and ingest documents into vector database."
    try:
        if not request.vector_db_index:
            raise HTTPException(status_code=400, detail="vector_db_index is required")

        files = request.files if isinstance(request.files, list) else [request.files]

        ingest_params = {
            "files": files,
            "vector_db_index": request.vector_db_index,
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
            "model": request.embedding_model,
            "index_schema": request.index_schema,
        }

        if request.json_data is not None:
            ingest_params["json_data"] = request.json_data

        success, message, processed_files, failed_files = (
            await document_service.ingest_documents(**ingest_params)
        )

        return DocumentIngestResponseSchema(
            success=success,
            message=message,
            processed_files=processed_files,
            failed_files=failed_files,
        )

    except Exception as ex:
        logger.error(f"[DocumentRoutes] Error in document ingestion: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing documents: {str(ex)}",
        )


@router.delete(
    "/api/documents/{index_name}", response_model=DocumentDeleteResponseSchema
)
async def delete_documents(
    index_name: str,
    document_service: DocumentService = Depends(get_document_service),
):
    "Delete documents from vector database."

    try:
        success, message, deleted_count = document_service.delete_documents(
            index_name=index_name
        )

        return DocumentDeleteResponseSchema(
            success=success, message=message, deleted_count=deleted_count
        )

    except Exception as ex:
        logger.error(f"[DocumentRoutes] Error deleting documents: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting documents: {str(ex)}",
        )


@router.post("/api/documents/ingest/positions")
async def ingest_positions(
    file_path: str,
    index_name: str,
    document_service: DocumentService = Depends(get_document_service),
):
    "Ingest positions from a file, flatten labels, and store in vector DB using index schema."

    try:

        success, message, processed_files, failed_files = (
            await document_service.ingest_positions(file_path, index_name)
        )

        return DocumentIngestResponseSchema(
            success=success,
            message=message,
            processed_files=processed_files,
            failed_files=failed_files,
        )

    except Exception as ex:
        logger.error(f"[DocumentRoutes] Error in positions ingestion: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing positions: {str(ex)}",
        )
