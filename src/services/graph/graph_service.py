import os
from typing import AsyncGenerator
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig

from src.schemas.graph_schema import (
    AgentState,
)
from src.services.data_api.app_settings import AppSettingsService
from src.services.logger.logger_service import LoggerService


class GraphService:
    """Service for handling multi-agent graph execution with supervisor pattern."""

    def __init__(self, app_settings_service: AppSettingsService, graph):
        self.logger = LoggerService().get_logger(__name__)
        self.app_settings_service = app_settings_service
        self.graph = graph

        self.redis_history_db = os.getenv("REDIS_HISTORY_DB")
        self.redis_url = f"redis://{os.getenv('REDIS_PASSWORD', '') + '@' if os.getenv('REDIS_PASSWORD') else ''}{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}{'/' + self.redis_history_db if self.redis_history_db else ''}"
        self.tracer_type = os.getenv("TRACER_TYPE", "langsmith")
        os.environ["REDIS_URL"] = self.redis_url

    async def execute_graph(
        self,
        user_input: str,
        app_id: int,
        user_id: str | None = None,
        context: dict | None = None,
        parameters: dict | None = None,
    ) -> str:
        """
        Execute the multi-agent graph solution with supervisor pattern.
        """
        self.logger.info(
            f"[GraphService|execute_graph] started (app_id={app_id}, user_id={user_id})"
        )
        try:
            graph_config = None
            if parameters and "graph_config" in parameters:
                graph_config = parameters["graph_config"]
            user_input = await self.graph.prepare_graph_execution(
                graph_config, user_input
            )

            initial_parameters = dict(parameters) if parameters else {}
            initial_parameters["app_id"] = app_id
            initial_state = AgentState(
                messages=[HumanMessage(content=user_input)],
                context=context or {},
                parameters=initial_parameters,
                user_id=user_id or uuid4(),
            )
            recursion_limit = getattr(self.graph.graph_config, "recursion_limit", 1)
            result = await self.graph.workflow.ainvoke(
                initial_state,
                RunnableConfig(recursion_limit=recursion_limit),
            )
            final_response = self.graph._generate_final_response(result)
            self.logger.info(
                f"[GraphService|execute_graph] finished (app_id={app_id}, user_id={user_id})"
            )
            return final_response
        except Exception as ex:
            self.logger.error(
                f"[GraphService|execute_graph] error (app_id={app_id}, user_id={user_id}): {str(ex)}"
            )
            return await self.graph._handle_execution_error(user_input, str(ex))

    async def execute_graph_stream(
        self,
        user_input: str,
        app_id: int,
        user_id: str | None = None,
        context: dict | None = None,
        parameters: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Execute the multi-agent graph solution and stream only the final response token by token.
        """
        self.logger.info(
            f"[GraphService|execute_graph_stream] started (app_id={app_id}, user_id={user_id})"
        )
        try:
            graph_config = None
            if parameters and "graph_config" in parameters:
                graph_config = parameters["graph_config"]
            user_input = await self.graph.prepare_graph_execution(
                graph_config, user_input
            )
            initial_state = AgentState(
                messages=[HumanMessage(content=user_input)],
                context=context or {},
                parameters=parameters or {},
            )
            recursion_limit = getattr(self.graph.graph_config, "recursion_limit", 1)
            result = await self.graph.workflow.ainvoke(
                initial_state,
                RunnableConfig(recursion_limit=recursion_limit),
            )
            final_response = self.graph._generate_final_response(result)
            async for token in self.graph._tokenize(final_response):
                yield token
            self.logger.info(
                f"[GraphService|execute_graph_stream] finished (app_id={app_id}, user_id={user_id})"
            )
        except Exception as ex:
            self.logger.error(
                f"[GraphService|execute_graph_stream] error (app_id={app_id}, user_id={user_id}): {str(ex)}"
            )
            self.logger.info(
                f"[GraphService|execute_graph_stream] finished (app_id={app_id}, user_id={user_id})"
            )
            error_response = await self.graph._handle_execution_error(
                user_input, str(ex)
            )
            async for token in self.graph._tokenize(error_response):
                yield token

    async def _tokenize(self, text: str) -> AsyncGenerator[str, None]:
        for word in text.split():
            yield word + " "
