import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from schemas.document_schema import (
    DocumentDeleteResponseSchema,
    DocumentIngestRequestSchema,
    DocumentIngestResponseSchema,
)
from services.document.document_service import DocumentService


def get_document_service():
    return DocumentService()


router = APIRouter(tags=["Documents"])


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

        if request.json_data is not None:
            success, message, processed_files, failed_files = (
                document_service.ingest_documents(
                    files=(
                        request.files
                        if isinstance(request.files, list)
                        else [request.files]
                    ),
                    vector_db_index=request.vector_db_index,
                    chunk_size=request.chunk_size,
                    chunk_overlap=request.chunk_overlap,
                    model=request.embedding_model,
                    index_schema=request.index_schema,
                    json_data=request.json_data,
                )
            )
        else:
            success, message, processed_files, failed_files = (
                await document_service.ingest_documents(
                    files=(
                        request.files
                        if isinstance(request.files, list)
                        else [request.files]
                    ),
                    vector_db_index=request.vector_db_index,
                    chunk_size=request.chunk_size,
                    chunk_overlap=request.chunk_overlap,
                    model=request.embedding_model,
                    index_schema=request.index_schema,
                )
            )

        return DocumentIngestResponseSchema(
            success=success,
            message=message,
            processed_files=processed_files,
            failed_files=failed_files,
        )

    except Exception as ex:
        logging.getLogger("logger").error(f"Error in document ingestion: {str(ex)}")
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
        success, message, deleted_count = await document_service.delete_documents(
            index_name=index_name
        )

        return DocumentDeleteResponseSchema(
            success=success, message=message, deleted_count=deleted_count
        )

    except Exception as ex:
        logging.getLogger("logger").error(f"Error deleting documents: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting documents: {str(ex)}",
        )


from fastapi import Query


@router.post("/api/documents/ingest/positions")
async def ingest_positions(
    file_path: str = Query(..., description="Path to positions JSON file"),
    index_name: str = Query(..., description="Redis index name"),
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
        logging.getLogger("logger").error(f"Error in positions ingestion: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing positions: {str(ex)}",
        )
