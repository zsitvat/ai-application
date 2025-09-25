from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.schemas.graph_schema import RestOperationPostSchema
from src.services.data_api.app_settings import AppSettingsService
from src.services.graph.graph import Graph
from src.services.graph.graph_service import GraphService
from src.services.logger.logger_service import LoggerService

logger = LoggerService().setup_logger()

router = APIRouter(tags=["graph"])


def get_app_settings_service() -> AppSettingsService:
    """Dependency injection for AppSettingsService.

    Returns:
        AppSettingsService: Instance of the application settings service
    """
    return AppSettingsService()


def get_graph_dependency(
    app_settings_service: AppSettingsService = Depends(get_app_settings_service),
) -> Graph:
    """Dependency injection for Graph instance.

    Args:
        app_settings_service: Application settings service dependency

    Returns:
        Graph: Configured graph instance
    """
    return Graph(
        graph_config=None,
        logger=LoggerService().setup_logger(),
        app_settings_service=app_settings_service,
    )


def get_graph_service(
    app_settings_service: AppSettingsService = Depends(get_app_settings_service),
    graph: Graph = Depends(get_graph_dependency),
) -> GraphService:
    """Dependency injection for GraphService.

    Args:
        app_settings_service: Application settings service dependency
        graph: Graph instance dependency

    Returns:
        GraphService: Configured graph service instance
    """
    return GraphService(app_settings_service, graph)


@router.post("/api/graph", response_model=str)
async def execute_graph(
    request: RestOperationPostSchema,
    graph_service: GraphService = Depends(get_graph_service),
) -> str:
    """Execute multi-agent graph solution.

    This endpoint executes a multi-agent workflow using the provided input
    and returns the final response as a complete string.

    Args:
        request: The graph execution request containing user input and parameters
        graph_service: Injected graph service for workflow execution

    Returns:
        str: Final response from the multi-agent graph execution

    Raises:
        HTTPException: 400 for invalid input, 503 for service unavailable,
                      500 for unexpected errors
    """
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

    except ValueError as ex:
        logger.error(f"[GraphRoutes] Invalid input in graph execution: {str(ex)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input for graph execution: {str(ex)}",
        )
    except RuntimeError as ex:
        logger.error(f"[GraphRoutes] Runtime error in graph execution: {str(ex)}")
        raise HTTPException(
            status_code=503,
            detail=f"Graph execution service unavailable: {str(ex)}",
        )
    except Exception as ex:
        logger.error(f"[GraphRoutes] Unexpected error in graph execution: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error executing multi-agent graph: {str(ex)}",
        )


@router.post("/api/graph/stream")
async def execute_graph_stream(
    request: RestOperationPostSchema,
    graph_service: GraphService = Depends(get_graph_service),
) -> StreamingResponse:
    """Execute multi-agent graph solution with token-by-token streaming.

    This endpoint executes a multi-agent workflow and streams the final
    response token by token for real-time user experience.

    Args:
        request: The graph execution request containing user input and parameters
        graph_service: Injected graph service for workflow execution

    Returns:
        StreamingResponse: Streaming response with tokens from graph execution

    Raises:
        HTTPException: 400 for invalid input, 503 for service unavailable,
                      500 for unexpected errors
    """
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

    except ValueError as ex:
        logger.error(f"[GraphRoutes] Invalid input in graph streaming: {str(ex)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input for graph streaming: {str(ex)}",
        )
    except RuntimeError as ex:
        logger.error(f"[GraphRoutes] Runtime error in graph streaming: {str(ex)}")
        raise HTTPException(
            status_code=503,
            detail=f"Graph streaming service unavailable: {str(ex)}",
        )
    except Exception as ex:
        logger.error(f"[GraphRoutes] Unexpected error in graph streaming: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error executing streaming multi-agent graph: {str(ex)}",
        )
