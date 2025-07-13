import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from services.data_api.app_settings import AppSettingsService
from services.graph.graph_service import GraphService
from services.graph.graph_studio_service import GraphStudioService

router = APIRouter(tags=["Graph Studio"])


def get_app_settings_service():
    return AppSettingsService()


def get_graph_service(
    app_settings_service: AppSettingsService = Depends(get_app_settings_service),
):
    return GraphService(app_settings_service)


def get_graph_studio_service(
    graph_service: GraphService = Depends(get_graph_service),
):
    return GraphStudioService(graph_service)


@router.get("/api/graph-studio/export/{app_id}")
async def export_graph_for_studio(
    app_id: int,
    parameters: dict | None = None,
    studio_service: GraphStudioService = Depends(get_graph_studio_service),
):
    """Export graph configuration for LangGraph Studio."""
    try:
        result = studio_service.export_graph_for_studio(app_id, parameters)
        return result

    except Exception as ex:
        logging.getLogger("logger").error(
            f"Error exporting graph for studio: {str(ex)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error exporting graph for studio: {str(ex)}",
        )


@router.post("/api/graph-studio/visualization/{app_id}")
async def save_graph_visualization(
    app_id: int,
    parameters: dict | None = None,
    output_path: str | None = Query(
        None, description="Custom output path for visualization file"
    ),
    studio_service: GraphStudioService = Depends(get_graph_studio_service),
):
    """Save graph visualization for debugging and studio integration."""
    try:
        result = studio_service.save_graph_visualization(
            app_id, parameters, output_path
        )
        return {"visualization_path": result}

    except Exception as ex:
        logging.getLogger("logger").error(
            f"Error saving graph visualization: {str(ex)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error saving graph visualization: {str(ex)}",
        )


@router.post("/api/graph-studio/config")
async def generate_langgraph_studio_config(
    config_path: str = Query(
        "langgraph.json", description="Path for the studio configuration file"
    ),
    studio_service: GraphStudioService = Depends(get_graph_studio_service),
):
    """Generate or update LangGraph Studio configuration file."""
    try:
        studio_service.generate_langgraph_studio_config(config_path)
        return {
            "message": f"LangGraph Studio configuration generated at: {config_path}"
        }

    except Exception as ex:
        logging.getLogger("logger").error(f"Error generating studio config: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating studio config: {str(ex)}",
        )
