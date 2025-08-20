from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.schemas.graph_schema import RestOperationPostSchema
from src.services.data_api.app_settings import AppSettingsService
from src.services.graph.graph_service import GraphService
from src.services.logger.logger_service import LoggerService

logger = LoggerService().setup_logger()

router = APIRouter(tags=["graph"])


def get_app_settings_service():
    return AppSettingsService()


def get_graph_service(
    app_settings_service: AppSettingsService = Depends(get_app_settings_service),
):
    return GraphService(app_settings_service)


@router.post("/api/graph", response_model=str)
async def execute_graph(
    request: RestOperationPostSchema,
    graph_service: GraphService = Depends(get_graph_service),
):
    """Execute multi-agent graph solution."""
    try:
        app_id = int(request.applicationIdentifier.applicationIdentifier)

        result = await graph_service.execute_graph(
            user_input=request.user_input,
            app_id=app_id,
            user_id=str(request.uuid),
            context=request.context,
            parameters=request.parameters,
        )

        return result

    except Exception as ex:
        logger.error(f"Error in graph execution: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing multi-agent graph: {str(ex)}",
        )


@router.post("/api/graph/stream")
async def execute_graph_stream(
    request: RestOperationPostSchema,
    graph_service: GraphService = Depends(get_graph_service),
):
    """Execute multi-agent graph solution with token-by-token streaming."""
    try:
        app_id = int(request.applicationIdentifier.applicationIdentifier)

        async def generate_stream():
            async for token in graph_service.execute_graph_stream(
                user_input=request.user_input,
                app_id=app_id,
                user_id=str(request.uuid),
                context=request.context,
                parameters=request.parameters,
            ):
                yield token

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as ex:
        logger.error(f"Error in graph streaming execution: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing streaming multi-agent graph: {str(ex)}",
        )
