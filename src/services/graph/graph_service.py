import os
from typing import AsyncGenerator
from uuid import uuid4
import re
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig

from src.config.app_config import config
from src.config.constants import DEFAULT_RECURSION_LIMIT
from src.schemas.graph_schema import AgentState
from src.services.data_api.app_settings import AppSettingsService
from src.services.graph.graph import Graph
from src.services.logger.logger_service import LoggerService


class GraphService:
    """Service for handling multi-agent graph execution with supervisor pattern.

    This service manages the execution of multi-agent workflows using a supervisor
    pattern for coordinating different agents in the system.

    Attributes:
        logger: Logger instance for debugging and monitoring
        app_settings_service: Service for managing application settings
        graph: Graph instance for workflow execution
        redis_history_db: Redis database for chat history
        redis_url: Connection URL for Redis
        tracer_type: Type of tracer for monitoring
    """

    def __init__(self, app_settings_service: AppSettingsService, graph: Graph) -> None:
        """Initialize the GraphService.

        Args:
            app_settings_service: Service for managing application settings
            graph: Graph instance for workflow execution
        """
        self.logger = LoggerService().setup_logger()
        self.app_settings_service = app_settings_service
        self.graph = graph

        self.redis_config = config.redis
        self.redis_url = config.redis.url
        self.tracer_type = config.tracing.tracer_type
        os.environ["REDIS_URL"] = self.redis_url

    def _clean_user_input(self, user_input: str) -> str:
        """Remove file upload notifications from user input before sending to AI.

        Args:
            user_input: The original user input that may contain file upload notifications

        Returns:
            str: Cleaned user input without file upload notifications
        """

        # Remove [File uploaded: ...] and [Image uploaded: ...] patterns
        cleaned_input = re.sub(r"\[(?:File|Image) uploaded: [^\]]+\]", "", user_input)
        # Clean up extra whitespace
        cleaned_input = " ".join(cleaned_input.split())
        return cleaned_input.strip()

    def _prepare_initial_state(
        self,
        user_input: str,
        app_id: int,
        user_id: str | None = None,
        context: dict[str, any] | None = None,
        parameters: dict[str, any] | None = None,
    ) -> AgentState:
        """Prepare the initial state for graph execution.

        Args:
            user_input: The user's input message
            app_id: Application identifier
            user_id: Optional user identifier, generates UUID if None
            context: Optional context dictionary for the conversation
            parameters: Optional parameters for graph execution

        Returns:
            AgentState: Prepared initial state for the workflow
        """
        initial_parameters = dict(parameters) if parameters else {}
        initial_parameters["app_id"] = app_id
        thread_id = user_id if user_id is not None else str(uuid4())
        initial_parameters["thread_id"] = thread_id

        cleaned_input = self._clean_user_input(user_input)

        initial_context = dict(context) if context else {}
        initial_context["thread_id"] = thread_id

        return AgentState(
            messages=[HumanMessage(content=cleaned_input)],
            context=initial_context,
            parameters=initial_parameters,
            user_id=user_id or uuid4(),
        )

    async def _execute_workflow(self, initial_state: AgentState) -> dict[str, any]:
        """Execute the workflow with the given initial state.

        Args:
            initial_state: The prepared initial state for workflow execution

        Returns:
            Dict[str, Any]: Result of the workflow execution
        """
        recursion_limit = getattr(
            self.graph.graph_config, "recursion_limit", DEFAULT_RECURSION_LIMIT
        )
        thread_id = None
        if hasattr(initial_state, "context") and "thread_id" in initial_state.context:
            thread_id = initial_state.context["thread_id"]
        elif (
            hasattr(initial_state, "parameters")
            and "thread_id" in initial_state.parameters
        ):
            thread_id = initial_state.parameters["thread_id"]
        return await self.graph.workflow.ainvoke(
            initial_state,
            RunnableConfig(recursion_limit=recursion_limit, thread_id=thread_id),
        )

    async def execute_graph(
        self,
        user_input: str,
        app_id: int,
        user_id: str | None = None,
        context: dict[str, any] | None = None,
        parameters: dict[str, any] | None = None,
    ) -> str:
        """Execute the multi-agent graph solution with supervisor pattern.

        Args:
            user_input: The user's input message
            app_id: Application identifier
            user_id: Optional user identifier
            context: Optional context dictionary for the conversation
            parameters: Optional parameters for graph execution

        Returns:
            str: Final response from the graph execution

        Raises:
            RuntimeError: When there's a runtime error in graph execution
            Exception: When there's an unexpected error
        """
        self.logger.debug(f"Executing graph for app_id={app_id}, user_id={user_id}")
        try:
            # Prepare graph configuration
            graph_config = parameters.get("graph_config") if parameters else None
            prepared_input = await self.graph.prepare_graph_execution(
                graph_config, user_input
            )

            # Execute workflow
            initial_state = self._prepare_initial_state(
                prepared_input, app_id, user_id, context, parameters
            )
            result = await self._execute_workflow(initial_state)

            final_response = self.graph._generate_final_response(result)
            return final_response
        except RuntimeError as ex:
            self.logger.error(
                f"Runtime error in graph execution (app_id={app_id}): {str(ex)}"
            )
            return await self.graph._handle_execution_error(user_input, str(ex))
        except Exception as ex:
            self.logger.error(
                f"Unexpected error in graph execution (app_id={app_id}): {str(ex)}"
            )
            return await self.graph._handle_execution_error(user_input, str(ex))

    async def execute_graph_stream(
        self,
        user_input: str,
        app_id: int,
        user_id: str | None = None,
        context: dict[str, any] | None = None,
        parameters: dict[str, any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Execute the multi-agent graph solution and stream the final response token by token.

        Args:
            user_input: The user's input message
            app_id: Application identifier
            user_id: Optional user identifier
            context: Optional context dictionary for the conversation
            parameters: Optional parameters for graph execution

        Yields:
            str: Tokens from the final response

        Raises:
            RuntimeError: When there's a runtime error in graph execution
            Exception: When there's an unexpected error
        """
        self.logger.debug(
            f"Streaming graph execution for app_id={app_id}, user_id={user_id}"
        )
        try:
            # Prepare graph configuration
            graph_config = parameters.get("graph_config") if parameters else None
            prepared_input = await self.graph.prepare_graph_execution(
                graph_config, user_input
            )

            # Execute workflow
            initial_state = self._prepare_initial_state(
                prepared_input, app_id, user_id, context, parameters
            )
            result = await self._execute_workflow(initial_state)

            final_response = self.graph._generate_final_response(result)
            async for token in self.graph._tokenize(final_response):
                yield token
        except RuntimeError as ex:
            self.logger.error(
                f"Runtime error in graph streaming (app_id={app_id}): {str(ex)}"
            )
            error_response = await self.graph._handle_execution_error(
                user_input, str(ex)
            )
            async for token in self.graph._tokenize(error_response):
                yield token
        except Exception as ex:
            self.logger.error(
                f"Unexpected error in graph streaming (app_id={app_id}): {str(ex)}"
            )
            error_response = await self.graph._handle_execution_error(
                user_input, str(ex)
            )
            async for token in self.graph._tokenize(error_response):
                yield token

    async def _tokenize(self, text: str) -> AsyncGenerator[str, None]:
        """Tokenize text by splitting into words and yielding them with spaces.

        Args:
            text: Text to tokenize

        Yields:
            str: Individual tokens with trailing spaces
        """
        for word in text.split():
            yield word + " "
