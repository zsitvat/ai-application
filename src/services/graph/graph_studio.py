from langchain_core.runnables.config import RunnableConfig

from src.services.graph.graph import Graph
from src.services.logger.logger_service import LoggerService

graph = Graph(LoggerService().get_logger(__name__), None, None)


async def get_compiled_workflow_for_studio(config: RunnableConfig):
    app_id: int = int(config.get("app_id", "0"))
    parameters = config.get("parameters")

    return await graph.get_compiled_workflow(app_id, parameters)
