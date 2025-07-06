import logging

from fastapi import APIRouter, Depends, HTTPException

from schemas.graph_schema import RestOperationPostSchema
from services.data_api.app_settings import AppSettingsService
from services.graph.graph_service import GraphService

router = APIRouter(tags=["Graph"])


def get_app_settings_service():
    return AppSettingsService()


def get_graph_service(
    app_settings_service: AppSettingsService = Depends(get_app_settings_service),
):
    return GraphService(app_settings_service)


@router.post("/api/graph", response_model=str)
async def execute_graph(
    request: RestOperationPostSchema,
    save_visualization: bool = False,
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
            save_visualization=save_visualization,
        )

        return result

    except Exception as ex:
        logging.getLogger("logger").error(f"Error in graph execution: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing multi-agent graph: {str(ex)}",
        )
