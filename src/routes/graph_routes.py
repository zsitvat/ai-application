from fastapi import APIRouter, HTTPException, Depends
import logging

from schemas.graph_schema import GraphRequestSchema, GraphResponseSchema
from services.graph.graph_service import GraphService

router = APIRouter()


def get_graph_service():
    return GraphService()


@router.post("/api/graph", response_model=GraphResponseSchema)
async def execute_graph(
    request: GraphRequestSchema,
    graph_service: GraphService = Depends(get_graph_service),
):
    "Execute multi-agent graph solution."
    try:
        result = await graph_service.execute_graph(
            user_input=request.user_input,
            user_id=str(request.user_id),
            context=request.context,
        )

        return GraphResponseSchema(result=result)

    except Exception as ex:
        logging.getLogger("logger").error(f"Error in graph execution: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing multi-agent graph: {str(ex)}",
        )
