from fastapi import APIRouter, HTTPException, Depends
import logging

from services.vector_db.vector_db_service import VectorDb
from schemas.vector_db_post_schema import VectorDbPostSchema
from schemas.response_schema import VectorDbResponseSchema

router = APIRouter()


def get_vector_db_service():
    return VectorDb()


@router.post("/api/vector_db/create", response_model=VectorDbResponseSchema)
async def create_vector_db(
    request: VectorDbPostSchema,
    vector_db_service: VectorDb = Depends(get_vector_db_service),
):
    """Create a vector database from a given file.

    Args:
        request (VectorDbPostSchema): Request body
        vector_db_service (VectorDb): VectorDb service instance

    Returns:
            VectorDbResponseSchema
    """

    try:

        if request.db_type == "deeplake":
            response = await vector_db_service.create_vector_db_deeplake(
                db_path=request.db_path,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                overwrite=request.overwrite,
                documents=request.documents,
                model=request.model,
                encoding=request.file_load_encoding,
                sheet_name=request.sheet_name,
            )

            return {"response": response}
        else:
            raise HTTPException(
                status_code=400, detail="Vector database type is not supported!"
            )

    except Exception as ex:
        logging.getLogger("logger").error(f"Error in vector_db/create route: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the request in vector_db/create: {str(ex)}",
        )
