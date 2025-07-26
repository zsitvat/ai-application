import asyncio
import importlib
import json
import logging
import os
from functools import partial
from typing import Any, AsyncGenerator, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from schemas.graph_schema import Agent, AgentState, CheckpointerType, GraphConfig, Model
from services.chat_history.redis_chat_history import RedisChatHistoryService
from services.data_api.app_settings import AppSettingsService
from services.data_api.chat_history import DataChatHistoryService
from services.graph.tools.tools_config import AVAILABLE_TOOLS
from services.validators.personal_data.personal_data_filter_checkpointer import (
    PersonalDataFilterCheckpointer,
)
from services.validators.personal_data.personal_data_filter_service import (
    PersonalDataFilterService,
)
from services.validators.topic_validator.topic_validator_service import (
    TopicValidatorService,
)
from utils.get_prompt import get_prompt_by_type
from utils.select_model import get_chat_model


class GraphService:
    """Service for handling multi-agent graph execution with supervisor pattern."""

    def __init__(self, app_settings_service: AppSettingsService):
        self.logger = logging.getLogger(__name__)
        self.app_settings_service = app_settings_service
        self.graph_config: GraphConfig | None = None
        self.workflow = None

        self.redis_history_db = os.getenv("REDIS_HISTORY_DB")
        self.redis_url = f"redis://{os.getenv('REDIS_PASSWORD') + '@' if os.getenv('REDIS_PASSWORD') else ''}{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}{'/' + self.redis_history_db if self.redis_history_db else ''}"
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

        Args:
            user_input (str): User input/question
            app_id (str): Application identifier for loading configurations
            user_id (str, optional): User identifier
            context (dict, optional): Additional context
            parameters (dict, optional): Runtime parameters including graph configurations

        Returns:
            str: Generated response from agents
        """

        self.logger.info(
            f"[GraphService|execute_graph] started (app_id={app_id}, user_id={user_id})"
        )
        try:
            user_input = await self._prepare_graph_execution(
                app_id, parameters, user_input
            )

            initial_state = AgentState(
                messages=[HumanMessage(content=user_input)],
                next="",
                user_input=user_input,
                context=context or {},
                parameters=parameters or {},
            )

            recursion_limit = getattr(self.graph_config, "recursion_limit", 1)

            result = await self.workflow.ainvoke(
                initial_state,
                {"recursion_limit": recursion_limit, "thread_id": user_id},
            )

            final_response = self._generate_final_response(result)

            self.logger.info(
                f"[GraphService|execute_graph] finished (app_id={app_id}, user_id={user_id})"
            )
            return final_response

        except Exception as ex:
            self.logger.error(
                f"[GraphService|execute_graph] error (app_id={app_id}, user_id={user_id}): {str(ex)}"
            )
            self.logger.info(
                f"[GraphService|execute_graph] finished (app_id={app_id}, user_id={user_id})"
            )
            return await self._handle_execution_error(user_input, str(ex))

    async def execute_graph_stream(
        self,
        user_input: str,
        app_id: int,
        user_id: str | None = None,
        context: dict | None = None,
        parameters: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Execute the multi-agent graph solution with token-by-token streaming output.
        Same as execute_graph but streams the response text token by token.

        Args:
            user_input (str): User input/question
            app_id (str): Application identifier for loading configurations
            user_id (str, optional): User identifier
            context (dict, optional): Additional context
            parameters (dict, optional): Runtime parameters including graph configurations

        Yields:
            str: Token chunks of the final response
        """

        self.logger.info(
            f"[GraphService|execute_graph_stream] started (app_id={app_id}, user_id={user_id})"
        )
        try:
            user_input = await self._prepare_graph_execution(
                app_id, parameters, user_input
            )

            initial_state = AgentState(
                messages=[HumanMessage(content=user_input)],
                next="",
                user_input=user_input,
                context=context or {},
                parameters=parameters or {},
            )

            recursion_limit = getattr(self.graph_config, "recursion_limit", 1)

            async for message_chunk, metadata in self.workflow.astream(
                initial_state,
                {"recursion_limit": recursion_limit, "thread_id": user_id},
                stream_mode="messages",
            ):
                if hasattr(message_chunk, "content") and message_chunk.content:
                    yield message_chunk.content

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
            error_response = await self._handle_execution_error(user_input, str(ex))
            yield error_response

    async def get_compiled_workflow(self, app_id: int, parameters: dict | None = None):
        """Get the compiled workflow for LangGraph Studio integration (async)."""

        self.logger.info(
            f"[GraphService|get_compiled_workflow] started (app_id={app_id})"
        )
        try:
            await self._load_graph_configuration(app_id, parameters)

            if not self.workflow:
                workflow_builder = self._build_workflow()
                checkpointer = await self._create_checkpointer()
                self.workflow = workflow_builder.compile(checkpointer=checkpointer)

            self.logger.info(
                f"[GraphService|get_compiled_workflow] finished (app_id={app_id})"
            )
            return self.workflow
        except Exception as ex:
            self.logger.error(
                f"[GraphService|get_compiled_workflow] error (app_id={app_id}): {str(ex)}"
            )
            self.logger.info(
                f"[GraphService|get_compiled_workflow] finished (app_id={app_id})"
            )
            raise

    async def _load_graph_configuration(
        self, app_id: int, parameters: dict[str, Any] | None = None
    ) -> None:
        """Load graph configuration from app settings and parameters."""

        try:
            graph_config_data = None

            if parameters and "graph_config" in parameters:
                graph_config_data = parameters["graph_config"]
            else:
                app_settings = await self.app_settings_service.get_app_settings(app_id)
                graph_config_data = app_settings.get("graph_config")
                if isinstance(graph_config_data, str):
                    graph_config_data = json.loads(graph_config_data)

            if graph_config_data:
                self.graph_config = GraphConfig(**graph_config_data)
            else:
                raise ValueError(
                    "No graph configuration found in app settings or parameters"
                )

            self.logger.debug(
                f"[GraphService] Loaded graph config with {len(self.graph_config.agents)} agents for app_id: {app_id}"
            )

        except Exception as ex:
            self.logger.error(
                f"[GraphService] Error loading graph configuration for app_id: {app_id}, error: {str(ex)}"
            )
            raise

    async def _agent_node(
        self, state: AgentState, agent_name: str, agent_config: Agent
    ) -> AgentState:
        """Handle agent node execution."""

        try:
            llm = get_chat_model(
                provider=agent_config.chain.model.provider.value,
                deployment=agent_config.chain.model.deployment,
                model=agent_config.chain.model.name,
            )

            prompt = await get_prompt_by_type(
                prompt_id=agent_config.chain.prompt_id, tracer_type=self.tracer_type
            )

            allowed_tools = {}
            if hasattr(agent_config, "tools") and isinstance(agent_config.tools, dict):
                for tool_name, tool_config in agent_config.tools.items():
                    if tool_name in AVAILABLE_TOOLS:
                        allowed_tools[tool_name] = tool_config

            chain = prompt | llm.bind_tools(allowed_tools)
            response = await chain.ainvoke({"messages": state["messages"]})

            tool_result_message = None
            tool_call = None
            tool_name = None
            tool_args = None

            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})

            elif response.additional_kwargs.get("function_call"):
                try:
                    fc = response.additional_kwargs["function_call"]
                    tool_name = fc.get("name")
                    tool_args = json.loads(fc.get("arguments", "{}"))
                except Exception:
                    tool_name = None
                    tool_args = None

            if tool_name and tool_name in allowed_tools:
                tool_func = allowed_tools[tool_name].get("function")
                if callable(tool_func):
                    try:

                        if asyncio.iscoroutinefunction(tool_func):
                            tool_result = await tool_func(**tool_args)
                        else:
                            tool_result = tool_func(**tool_args)
                        tool_result_message = AIMessage(
                            content=f"Tool '{tool_name}' result: {tool_result}"
                        )
                    except Exception as tool_ex:
                        tool_result_message = AIMessage(
                            content=f"Tool '{tool_name}' error: {tool_ex}"
                        )

            state["messages"] = add_messages(state["messages"], [response])
            if tool_result_message:
                state["messages"] = add_messages(
                    state["messages"], [tool_result_message]
                )

            self.logger.debug(
                f"[GraphService] Agent node '{agent_name}' completed processing."
            )
            return state

        except Exception as ex:
            self.logger.error(
                f"[GraphService] Error in agent node '{agent_name}': {str(ex)}"
            )

            error_message = AIMessage(
                content=f"Agent {agent_name} encountered an error: {str(ex)}"
            )
            state["messages"] = add_messages(state["messages"], [error_message])
            return state

    def _build_supervisor_prompt(
        self, available_options: list[str], last_agent=None
    ) -> str:
        """Build the system prompt for the supervisor."""

        agent_descriptions = []
        for name, agent in self.graph_config.agents.items():
            if agent.enabled:
                agent_descriptions.append(f"- {name}")

        system_prompt_ending = ""
        if self.graph_config.allow_supervisor_finish:
            system_prompt_ending = "Based on the user input and conversation history, decide which agent should handle this next or if the task is complete (FINISH)."
        else:
            system_prompt_ending = "Based on the user input and conversation history, decide which agent should handle this next. You must select one of the available agents."

        last_agent_str = (
            f"\nThe last agent selected was: {last_agent}."
            if last_agent
            else "No last agent selected."
        )

        return f"""You are a supervisor managing a team of AI agents. Your job is to decide which agent should handle the user's request{"" if not self.graph_config.allow_supervisor_finish else " or if the task is complete"}.

Available agents and their capabilities:
{chr(10).join(agent_descriptions)}

Important: Avoid recursion. Do not select the same agent repeatedly in a loop, so check the conversation history.
If the same agent was just called, prefer choosing a different agent or FINISH if appropriate.
If the answer from the last agent was not satisfactory, you can choose a different agent to try again, but do not choose the same one.
{last_agent_str}

{system_prompt_ending}

Select one of: {available_options}"""

    def _build_function_definition(self, available_options: list[str]) -> dict:
        """Build the function definition for supervisor routing."""
        return {
            "name": "route",
            "description": "Select the next agent to handle the user's request.",
            "parameters": {
                "title": "routeSchema",
                "type": "object",
                "properties": {
                    "chain": {
                        "title": "Chain",
                        "anyOf": [
                            {"enum": available_options},
                        ],
                    },
                },
                "required": ["chain"],
            },
        }

    def _extract_next_agent_from_response(self, response) -> str | None:
        """Extract the next agent from LLM response."""

        if response.additional_kwargs.get("function_call"):
            return json.loads(
                response.additional_kwargs["function_call"]["arguments"]
            ).get("chain", None)
        elif response.tool_calls:
            return response.tool_calls[0].get("args", {}).get("chain", None)
        else:
            return None

    def _create_supervisor_node(self):
        """Create the supervisor node that decides which agent to call next."""

        async def supervisor_node(state: AgentState) -> AgentState:
            try:
                llm = get_chat_model(
                    provider=self.graph_config.supervisor.chain.model.provider.value,
                    deployment=self.graph_config.supervisor.chain.model.deployment,
                    model=self.graph_config.supervisor.chain.model.name,
                )

                enabled_agents = [
                    name
                    for name, agent in self.graph_config.agents.items()
                    if (agent.enabled and isinstance(agent, Agent))
                ]

                if not enabled_agents:
                    raise ValueError(
                        "No enabled agents available and supervisor finish is disabled"
                    )

                available_options = enabled_agents[:]
                if self.graph_config.allow_supervisor_finish:
                    available_options.append("FINISH")

                function_def = self._build_function_definition(available_options)
                system_prompt = self._build_supervisor_prompt(
                    available_options, state.get("last_agent")
                )

                prompt = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(content=system_prompt),
                        MessagesPlaceholder(variable_name="messages"),
                        (
                            "system",
                            "Given the conversation above, who should act next? Select one of: {options}",
                        ),
                    ]
                ).partial(options=str(available_options))

                chain = prompt | llm.bind_functions(
                    functions=[function_def], function_call="route"
                )

                response = await chain.ainvoke({"messages": state["messages"]})

                next_agent = self._extract_next_agent_from_response(response)

                if next_agent not in available_options:
                    if self.graph_config.allow_supervisor_finish:
                        next_agent = "FINISH"
                    else:
                        next_agent = enabled_agents[0]

                state["next"] = next_agent

                agent_name = None
                if response.additional_kwargs.get("function_call"):
                    try:
                        agent_name = json.loads(
                            response.additional_kwargs["function_call"]["arguments"]
                        ).get("chain", None)
                    except Exception:
                        agent_name = None
                elif response.tool_calls:
                    agent_name = (
                        response.tool_calls[0].get("args", {}).get("chain", None)
                    )
                state["last_agent"] = agent_name

                self.logger.debug(
                    f"[GraphService] Supervisor node decided next action: {state['next']}. State: {state}"
                )
                return state

            except Exception as ex:
                self.logger.error(
                    f"[GraphService] Error in supervisor node: {str(ex)}. State: {state}"
                )
                raise

        return supervisor_node

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with supervisor pattern."""

        workflow = StateGraph(AgentState)

        topic_validator_config = getattr(self.graph_config, "topic_validator", None)
        if topic_validator_config and getattr(topic_validator_config, "enabled", False):
            workflow.add_node("topic_validator", self._topic_validator_node)

        workflow.add_node("supervisor", self._create_supervisor_node())

        if self.graph_config.exception_chain:
            workflow.add_node("exception_chain", self._exception_chain_node)

        enabled_agents = {
            name: agent
            for name, agent in self.graph_config.agents.items()
            if agent.enabled and isinstance(agent, Agent)
        }

        if not enabled_agents and not self.graph_config.allow_supervisor_finish:
            raise ValueError(
                "No enabled agents available and supervisor finish is disabled"
            )

        for agent_name, agent_config in enabled_agents.items():
            agent_node = partial(
                self._agent_node, agent_name=agent_name, agent_config=agent_config
            )
            workflow.add_node(agent_name, agent_node)

        for agent_name in enabled_agents.keys():
            workflow.add_edge(agent_name, "supervisor")

        if self.graph_config.exception_chain:
            workflow.add_edge("exception_chain", END)

        conditional_mapping = {
            agent_name: agent_name for agent_name in enabled_agents.keys()
        }

        if self.graph_config.allow_supervisor_finish:
            conditional_mapping["FINISH"] = END

        workflow.add_conditional_edges(
            "supervisor",
            self.should_continue_from_supervisor,
            conditional_mapping,
        )

        topic_conditional_mapping = {
            "supervisor": "supervisor",
            "exception_chain": "exception_chain",
            "FINISH": END,
        }

        if topic_validator_config and getattr(topic_validator_config, "enabled", False):
            workflow.add_conditional_edges(
                "topic_validator",
                self.should_continue_from_topic_validator,
                topic_conditional_mapping,
            )
            workflow.set_entry_point("topic_validator")
        else:
            workflow.set_entry_point("supervisor")

        return workflow

    async def _create_checkpointer(self):
        """Create checkpointer based on configuration (async version)."""

        checkpointer_type = self.graph_config.checkpointer_type
        base_checkpointer = None

        if checkpointer_type == CheckpointerType.MEMORY:
            base_checkpointer = InMemorySaver()
        elif checkpointer_type == CheckpointerType.REDIS:
            try:
                if (
                    not hasattr(self, "_async_redis_saver")
                    or self._async_redis_saver is None
                ):
                    async with AsyncRedisSaver.from_conn_string(
                        redis_url=self.redis_url
                    ) as saver:
                        await saver.asetup()
                        self._async_redis_saver = saver
                base_checkpointer = self._async_redis_saver
            except Exception as ex:
                self.logger.warning(
                    f"[GraphService] Failed to create AsyncRedisSaver: {ex}, falling back to memory"
                )
                base_checkpointer = InMemorySaver()
        elif checkpointer_type == CheckpointerType.DATA:
            base_checkpointer = DataChatHistoryService()
        elif checkpointer_type == CheckpointerType.CUSTOM:
            if hasattr(self.graph_config, "custom_checkpointer_class"):
                module_path, class_name = (
                    self.graph_config.custom_checkpointer_class.rsplit(".", 1)
                )
                module = importlib.import_module(module_path)
                checkpointer_class = getattr(module, class_name)
                base_checkpointer = checkpointer_class()
            else:
                base_checkpointer = RedisChatHistoryService()
        else:
            self.logger.warning(
                f"[GraphService] Unknown checkpointer type: {checkpointer_type}, falling back to memory"
            )
            base_checkpointer = InMemorySaver()

        # Wrap with personal data filter if configured
        personal_data_config = self._find_personal_data_filter_config()
        if personal_data_config:
            personal_data_service = PersonalDataFilterService()
            return PersonalDataFilterCheckpointer(
                base_checkpointer=base_checkpointer,
                personal_data_service=personal_data_service,
                personal_data_config=personal_data_config,
                logger=self.logger,
            )

        return base_checkpointer

    async def _prepare_graph_execution(
        self, app_id: int, parameters: dict | None, user_input: str
    ) -> str:
        """
        Prepare graph for execution by loading config and validating input.
        If the input exceeds max length, it will be truncated.
        """

        await self._load_graph_configuration(app_id, parameters)

        if not self.graph_config or not self.graph_config.agents:
            raise ValueError("No valid graph configuration or agents found")

        if (
            self.graph_config.max_input_length > 0
            and len(user_input) > self.graph_config.max_input_length
        ):
            user_input = user_input[: self.graph_config.max_input_length]
            self.logger.debug(
                f"[GraphService] User input truncated to {self.graph_config.max_input_length} characters for app_id: {app_id}"
            )

        if not self.workflow:
            workflow_builder = self._build_workflow()
            checkpointer = await self._create_checkpointer()
            self.workflow = workflow_builder.compile(checkpointer=checkpointer)

        return user_input

    def _generate_final_response(self, result: AgentState) -> str:
        """Generate the final response from agent results."""

        final_response = None
        if result["messages"]:
            for msg in reversed(result["messages"]):
                if hasattr(msg, "content") and msg.content:
                    final_response = msg.content
                    break
            if final_response is None:
                final_response = str(result["messages"][-1])
        else:
            final_response = "No response generated from agents."

        self.logger.debug(
            "[GraphService] Graph execution completed successfully. Final result: %s",
            result,
        )
        return final_response

    async def _handle_execution_error(self, user_input: str, error_message: str) -> str:
        """Handle execution errors, potentially using exception chain."""

        if self.graph_config and self.graph_config.exception_chain:
            try:
                return await self._handle_exception_with_chain(
                    user_input, error_message
                )
            except Exception as chain_error:
                self.logger.error(
                    f"[GraphService] Exception chain also failed for user_input: {user_input}, error: {str(chain_error)}"
                )

        return f"Error executing multi-agent workflow: {error_message}"

    async def _handle_exception_with_chain(
        self, user_input: str, error_message: str
    ) -> str:
        """Handle exceptions using the configured exception chain."""

        try:
            llm = get_chat_model(
                provider=self.graph_config.exception_chain.chain.model.provider.value,
                deployment=self.graph_config.exception_chain.chain.model.deployment,
                model=self.graph_config.exception_chain.chain.model.name,
            )

            prompt = await get_prompt_by_type(
                prompt_id=self.graph_config.exception_chain.chain.prompt_id,
                tracer_type=self.tracer_type,
            )
            context = f"User input: {user_input}\nError: {error_message}"
            prompt = prompt.partial(context=context)

            chain = prompt | llm
            response = await chain.ainvoke(user_input)

            self.logger.debug(
                f"[GraphService] Exception handled by exception chain for user_input: {user_input}, error: {error_message}"
            )
            return response.content

        except Exception as ex:
            self.logger.error(
                f"[GraphService] Exception chain failed for user_input: {user_input}, error: {str(ex)}"
            )
            raise

    def _find_topic_validation_config(self):
        """Return topic validator config if enabled, else None."""
        config = getattr(self.graph_config, "topic_validator", None)
        if config and getattr(config, "enabled", False):
            return config
        return None

    def _handle_invalid_topic(self, state: AgentState, reason: str) -> AgentState:
        """Handle when topic validation fails."""

        if self.graph_config.exception_chain:
            state["next"] = "exception_chain"
            state["error_context"] = f"Topic validation failed: {reason}"
            self.logger.debug(
                f"[GraphService] Question rejected by topic validator, routing to exception chain. Reason: {reason}, State: {state}"
            )
        else:
            error_message = AIMessage(
                content="I can only help with work-related topics. Please ask questions about careers, job search, professional development, or workplace matters."
            )
            state["messages"] = add_messages(state["messages"], [error_message])
            state["next"] = "FINISH"
            self.logger.debug(
                f"[GraphService] Question rejected by topic validator, no exception chain available. Reason: {reason}, State: {state}"
            )
        return state

    def _find_personal_data_filter_config(self):
        """Return personal data filter config if enabled, else None."""
        config = getattr(self.graph_config, "personal_data_filter", None)
        if config and getattr(config, "enabled", False):
            return config
        return None

    async def _topic_validator_node(self, state: AgentState) -> AgentState:
        """Topic validator node logic."""

        try:
            topic_validation_config = self._find_topic_validation_config()

            if not topic_validation_config:
                self.logger.debug(
                    "[GraphService] Topic validation is disabled, skipping validation. State: %s",
                    state,
                )
                return state

            if not state["user_input"]:
                self.logger.warning(
                    f"[GraphService] No user input found for topic validation. State: {state}"
                )
                return state

            topic_validator_service = TopicValidatorService()

            allowed_topics = getattr(topic_validation_config, "allowed_topics", None)
            invalid_topics = getattr(topic_validation_config, "invalid_topics", None)
            model_config: Model = getattr(topic_validation_config, "model", None)

            if not model_config:
                raise ValueError("Topic validation model configuration is required")

            is_valid, topic, reason = await topic_validator_service.validate_topic(
                question=state["user_input"],
                model_provider=model_config.provider.value,
                model_name=model_config.name,
                model_deployment=model_config.deployment,
                allowed_topics=allowed_topics,
                invalid_topics=invalid_topics,
                raise_on_invalid=False,
            )

            self.logger.debug(
                f"[GraphService] Topic validation result: valid={is_valid}, topic={topic}, reason={reason}, State: {state}"
            )

            if not is_valid:
                return self._handle_invalid_topic(state, reason)

            self.logger.debug(
                f"[GraphService] Topic validation passed, proceeding to supervisor. State: {state}"
            )
            return state

        except Exception as ex:
            self.logger.error(
                f"[GraphService] Error in topic validator: {str(ex)}. State: {state}"
            )
            if self.graph_config.exception_chain:
                state["next"] = "exception_chain"
                state["error_context"] = f"Topic validation error: {str(ex)}"
                return state
            return state

    async def _exception_chain_node(self, state: AgentState) -> AgentState:
        """Exception chain node for handling errors and invalid inputs."""
        error_context = state.get("error_context", "General error occurred")
        try:
            llm = get_chat_model(
                provider=self.graph_config.exception_chain.chain.model.provider.value,
                deployment=self.graph_config.exception_chain.chain.model.deployment,
                model=self.graph_config.exception_chain.chain.model.name,
            )
        except Exception as ex:
            self.logger.error(
                f"[GraphService] Error getting LLM for exception chain: {str(ex)}. State: {state}"
            )
            return self._handle_exception_chain_fallback(state)

        prompt = None
        try:
            prompt = await get_prompt_by_type(
                prompt_id=self.graph_config.exception_chain.chain.prompt_id,
                tracer_type=self.tracer_type,
            )
            context = f"User input: {state['user_input']}\nContext: {error_context}"
            prompt = prompt.partial(context=context)
        except Exception as e:
            self.logger.error(
                f"Failed to load exception chain prompt {self.graph_config.exception_chain.chain.prompt_id} from {self.tracer_type}: {str(e)}"
            )
            return self._handle_exception_chain_fallback(state)

        try:
            chain = prompt | llm
            response = await chain.ainvoke({"messages": state["messages"]})
            state["messages"] = add_messages(state["messages"], [response])
            state["next"] = "FINISH"
            self.logger.debug(
                f"[GraphService] Exception chain node handled the error. State: {state}"
            )
            return state
        except Exception as ex:
            self.logger.error(
                f"[GraphService] Error in exception chain node: {str(ex)}. State: {state}"
            )
            return self._handle_exception_chain_fallback(state)

    def _handle_exception_chain_fallback(self, state: AgentState) -> AgentState:
        error_message = AIMessage(
            content="I apologize, but I'm unable to process your request at this time. Please try again later."
        )
        state["messages"] = add_messages(state["messages"], [error_message])
        state["next"] = "FINISH"
        return state

    def should_continue_from_supervisor(
        self, state: AgentState
    ) -> Literal["FINISH"] | str:
        """Determine next step after supervisor node."""

        if state["next"] == "FINISH":
            return "FINISH"
        return state["next"]

    def should_continue_from_topic_validator(
        self, state: AgentState
    ) -> Literal["supervisor", "exception_chain", "FINISH"]:
        """Determine next step after topic validator node."""

        next_step = state.get("next")
        if next_step == "exception_chain":
            return "exception_chain"
        elif next_step == "FINISH":
            return "FINISH"
        else:
            return "supervisor"
