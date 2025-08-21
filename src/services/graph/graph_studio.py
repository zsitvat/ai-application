from langchain_core.runnables.config import RunnableConfig

from src.services.data_api.app_settings import AppSettingsService
from src.services.graph.graph_service import GraphService

graph_service = GraphService(AppSettingsService())


async def get_compiled_workflow_for_studio(config: RunnableConfig):
    app_id: str = str(config.get("app_id", "0"))
    parameters = config.get("parameters")
    return await graph_service.get_compiled_workflow(int(app_id), parameters)
