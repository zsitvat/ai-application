import asyncio
import importlib
import json
import os
from functools import partial
from typing import Any, AsyncGenerator, Literal
from uuid import uuid4

import aiohttp
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.config import RunnableConfig

from src.schemas.graph_schema import (
    Agent,
    AgentState,
    CheckpointerType,
    GraphConfig,
    Model,
)
from src.services.chat_history.redis_chat_history import RedisChatHistoryService
from src.services.data_api.app_settings import AppSettingsService
from src.services.data_api.chat_history import DataChatHistoryService
from src.services.graph.tools.tools_config import AVAILABLE_TOOLS
from src.services.validators.personal_data.personal_data_filter_checkpointer import (
    PersonalDataFilterCheckpointer,
)
from src.services.validators.personal_data.personal_data_filter_service import (
    PersonalDataFilterService,
)
from src.services.validators.topic_validator.topic_validator_service import (
    TopicValidatorService,
)
from src.utils.get_prompt import get_prompt_by_type
from src.utils.select_model import get_chat_model
from src.services.logger.logger_service import LoggerService


class GraphService:
    """Service for handling multi-agent graph execution with supervisor pattern."""

    def __init__(self, app_settings_service: AppSettingsService):

        self.logger = LoggerService().get_logger(__name__)
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
                context=context or {},
                parameters=parameters or {},
                user_id=user_id or uuid4(),
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
        Execute the multi-agent graph solution and stream only the final response token by token.
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
                context=context or {},
                parameters=parameters or {},
            )

            recursion_limit = getattr(self.graph_config, "recursion_limit", 1)

            result = await self.workflow.ainvoke(
                initial_state,
                {"recursion_limit": recursion_limit, "thread_id": user_id},
            )

            final_response = self._generate_final_response(result)

            async for token in self._tokenize(final_response):
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
            error_response = await self._handle_execution_error(user_input, str(ex))
            async for token in self._tokenize(error_response):
                yield token

    async def _tokenize(self, text: str) -> AsyncGenerator[str, None]:
        for word in text.split():
            yield word + " "

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
        """Load graph configuration from parameters, app settings, or file path in .env as fallback."""

        try:
            graph_config_data = None

            # 1. Try parameters
            if parameters:
                if "graph_config" in parameters:
                    graph_config_data = parameters["graph_config"]
                    self.graph_config = GraphConfig(**graph_config_data)
                    self.logger.debug(
                        f"[GraphService] Loaded graph config from parameters for app_id: {app_id}"
                    )
                    return
                elif "config" in parameters:
                    graph_config_data = parameters["config"]
                    self.graph_config = GraphConfig(**graph_config_data)
                    self.logger.debug(
                        f"[GraphService] Loaded graph config from parameters for app_id: {app_id}"
                    )
                    return

            # 2. Try app settings (with error handling)
            try:
                app_settings = await self.app_settings_service.get_app_settings(app_id)
                graph_config_data = app_settings.get("graph_config")
                if isinstance(graph_config_data, str):
                    graph_config_data = json.loads(graph_config_data)

                if graph_config_data:
                    self.graph_config = GraphConfig(**graph_config_data)
                    self.logger.debug(
                        f"[GraphService] Loaded graph config with {len(self.graph_config.agents)} agents for app_id: {app_id}"
                    )
                    return
            except Exception as app_settings_ex:
                self.logger.warning(
                    f"[GraphService] Failed to load app settings for app_id: {app_id}, error: {str(app_settings_ex)}. Will attempt fallback."
                )

            # 3. Fallback: try file from .env
            graph_config_path = os.getenv("GRAPH_CONFIG_PATH")
            if graph_config_path:
                if os.path.isfile(graph_config_path):

                    def _read_json_file(path):
                        with open(path, "r", encoding="utf-8") as f:
                            return json.load(f)

                    graph_config_data = await asyncio.to_thread(
                        _read_json_file, graph_config_path
                    )
                    self.graph_config = GraphConfig(**graph_config_data)
                    self.logger.warning(
                        f"[GraphService] Loaded graph config from fallback file {graph_config_path} for app_id: {app_id}"
                    )
                    return
                else:
                    self.logger.warning(
                        f"[GraphService] GRAPH_CONFIG_PATH is set but file does not exist: {graph_config_path}"
                    )

            # 4. If nothing found, raise
            raise ValueError(
                "No graph configuration found in parameters, app settings, or fallback file"
            )

        except Exception as ex:
            self.logger.error(
                f"[GraphService] Error loading graph configuration for app_id: {app_id}, error: {str(ex)}"
            )
            raise

    async def _applicant_attributes_extractor_node(
        self, state: AgentState
    ) -> AgentState:
        """Node that extracts applicant attributes from all messages using LLM and updates application_attributes."""

        try:
            llm = get_chat_model(
                provider=self.graph_config.supervisor.chain.model.provider.value,
                deployment=self.graph_config.supervisor.chain.model.deployment,
                model=self.graph_config.supervisor.chain.model.name,
            )

            extract_prompt = (
                "Extract applicant data from a job application conversation. "
                "Based on the entire conversation, return a JSON object with the following fields: "
                "- applicant_name: applicant's name (required) "
                "- phone_number: phone number (required) "
                "- email: email address "
                "- position_name: position name (required) "
                "- position_id: position identifier (required) "
                "- application_reason: why they applied for the job (required) "
                "- language_skills: language skills "
                "- experience: experience (required) "
                "- other_information: other information "
                "If a field is not available, leave it empty. Return only the JSON object, nothing else."
            )

            messages_text = "\n".join(
                [
                    f"{msg.__class__.__name__}: {getattr(msg, 'content', str(msg))}"
                    for msg in state.messages
                ]
            )
            full_prompt = f"{extract_prompt}\n\nConversation history:\n{messages_text}"

            response = await llm.ainvoke(full_prompt)

            try:
                extracted = json.loads(response.content)
                if isinstance(extracted, dict):
                    for key, value in extracted.items():
                        if key in state.application_attributes and value:
                            state.application_attributes[key] = str(value)

                    # Check if all required fields are present and call endpoint if so
                    if self._check_required_fields_complete(
                        state.application_attributes
                    ):
                        await self._submit_application_data(
                            state.application_attributes
                        )

            except json.JSONDecodeError:
                self.logger.warning(
                    f"[GraphService|_applicant_attributes_extractor_node] Failed to parse JSON response: {response.content}"
                )

            return state

        except Exception as ex:
            self.logger.error(
                f"[GraphService|_applicant_attributes_extractor_node] error: {str(ex)}"
            )
            return state
        return state

    def _check_required_fields_complete(self, application_attributes: dict) -> bool:
        """Check if all required application attributes are present and not empty."""
        required_fields = [
            "applicant_name",
            "phone_number",
            "position_name",
            "position_id",
            "application_reason",
            "experience",
        ]

        for field in required_fields:
            value = application_attributes.get(field, "").strip()
            if not value:
                self.logger.debug(
                    f"[GraphService] Required field '{field}' is missing or empty"
                )
                return False

        self.logger.debug("[GraphService] All required fields are complete")
        return True

    async def _submit_application_data(self, application_attributes: dict) -> None:
        """Submit application data to the configured endpoint."""
        base_url = os.getenv("DATA_API_BASE_URL", "").rstrip("/")
        endpoint_path = os.getenv("APPLICATION_SUBMIT_ENDPOINT", "")

        if not base_url or not endpoint_path:
            self.logger.warning(
                "[GraphService] DATA_API_BASE_URL or APPLICATION_SUBMIT_ENDPOINT not configured in .env"
            )
            return

        endpoint_url = f"{base_url}/{endpoint_path.lstrip('/')}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint_url,
                    json=application_attributes,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        self.logger.info(
                            f"[GraphService] Successfully submitted application data to {endpoint_url}"
                        )
                    else:
                        response_text = await response.text()
                        self.logger.error(
                            f"[GraphService] Failed to submit application data. Status: {response.status}, Response: {response_text}"
                        )
        except Exception as ex:
            self.logger.error(
                f"[GraphService] Error submitting application data to {endpoint_url}: {str(ex)}"
            )

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

            prompt = self._inject_tool_info_into_prompt(prompt, agent_config)

            prompt_with_messages = ChatPromptTemplate.from_messages(
                [prompt, MessagesPlaceholder(variable_name="messages")]
            )

            allowed_tool_names = self._get_allowed_tool_names(agent_config)
            tools_to_bind = [AVAILABLE_TOOLS[name] for name in allowed_tool_names]
            chain = prompt_with_messages | llm.bind_tools(
                tools_to_bind, tool_choice=agent_config.tool_choice
            )

            prompt_context = {"messages": state.messages}

            if state.context:
                prompt_context.update(state.context)
            if state.parameters:
                prompt_context.update(state.parameters)

            prompt_context["application_attributes"] = state.application_attributes

            if "user_information" not in prompt_context:
                prompt_context["user_information"] = ""
            if "labels" not in prompt_context:
                prompt_context["labels"] = []
            if "positions" not in prompt_context:
                prompt_context["positions"] = []

            # Check for required tools and execute them first
            required_tools_executed = await self._execute_required_tools(agent_config)

            self._update_context_with_tool_results(
                prompt_context, required_tools_executed, state.application_attributes
            )

            self.logger.debug(
                f"[GraphService] Agent '{agent_name}' application_attributes after required tools: {state.application_attributes}"
            )
            self.logger.debug(
                f"[GraphService] Agent '{agent_name}' prompt_context keys after required tools: {list(prompt_context.keys())}"
            )

            response = await chain.ainvoke(prompt_context)

            tool_name, tool_args, tool_call_id = self._extract_tool_call(response)
            tool_func = self._find_tool_func(tool_name, tools_to_bind)

            state["messages"] = add_messages(state["messages"], [response])

            if tool_name and tool_func:
                self.logger.debug(
                    f"[GraphService] Original tool args from LLM for '{tool_name}': {tool_args}"
                )

                if "input_fields" in tool_args and isinstance(
                    tool_args["input_fields"], dict
                ):
                    search_values = tool_args["input_fields"]
                    tool_args = search_values.copy()

                    if tool_name in agent_config.tools:
                        config_input_fields = agent_config.tools[tool_name].get(
                            "input_fields", []
                        )
                        if config_input_fields:
                            mapped_args = {}
                            for field_name in config_input_fields:
                                if field_name in search_values:
                                    mapped_args[field_name] = search_values[field_name]
                                elif (
                                    field_name.startswith("labels_")
                                    and field_name[7:] in search_values
                                ):
                                    mapped_args[field_name] = search_values[
                                        field_name[7:]
                                    ]
                            tool_args = mapped_args

                if tool_name in agent_config.tools and isinstance(tool_args, dict):
                    tool_args = self._merge_tool_args_with_config(
                        tool_args, agent_config.tools[tool_name]
                    )
                    self.logger.debug(
                        f"[GraphService] Tool args after merge for '{tool_name}': {tool_args}"
                    )

                tool_result_message = await self._execute_tool_function(
                    tool_func, tool_name, tool_args, agent_config
                )

                if tool_result_message:
                    if isinstance(tool_result_message, ToolMessage):
                        tool_result_message.tool_call_id = (
                            tool_call_id or tool_name or "unknown_tool"
                        )

                    state["messages"] = add_messages(
                        state["messages"], [tool_result_message]
                    )

                    var_name = None
                    if (
                        hasattr(agent_config, "tools")
                        and tool_name in agent_config.tools
                    ):
                        var_name = agent_config.tools[tool_name].get("variable_name")

                    self._update_context_with_tool_results(
                        prompt_context,
                        [
                            {
                                "tool_name": tool_name,
                                "message": tool_result_message,
                                "variable_name": var_name,
                            }
                        ],
                        state.application_attributes,
                    )

                    prompt_context["messages"] = self._filter_problematic_messages(
                        state.messages
                    )

                    follow_up_response = await chain.ainvoke(prompt_context)
                    state["messages"] = add_messages(
                        state["messages"], [follow_up_response]
                    )

                    self.logger.debug(
                        f"[GraphService] Agent '{agent_name}' generated follow-up response after tool execution: {str(follow_up_response)[:100]}..."
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

    def _inject_tool_info_into_prompt(self, prompt, agent_config):
        if hasattr(agent_config, "tools") and isinstance(agent_config.tools, dict):
            all_tool_info = []
            for tool_name, tool_conf in agent_config.tools.items():
                tool_info_lines = [f"Tool: {tool_name}"]
                for k, v in tool_conf.items():
                    if isinstance(v, (list, tuple)):
                        tool_info_lines.append(f"{k}: {', '.join(map(str, v))}")
                    else:
                        tool_info_lines.append(f"{k}: {v}")
                all_tool_info.append("\n".join(tool_info_lines))
            if all_tool_info:
                prompt = prompt.partial(agent_tool_info="\n\n".join(all_tool_info))
        return prompt

    def _get_allowed_tool_names(self, agent_config):
        allowed_tool_names = []
        if hasattr(agent_config, "tools") and isinstance(agent_config.tools, dict):
            for tool_name, tool_config in agent_config.tools.items():
                # Skip required tools
                if tool_config.get("required", False):
                    continue
                if tool_name in AVAILABLE_TOOLS:
                    allowed_tool_names.append(tool_name)
        return allowed_tool_names

    def _extract_tool_call(self, response):
        tool_name = None
        tool_args = None
        tool_call_id = None

        if hasattr(response, "tool_calls") and response.tool_calls:
            try:
                tool_call = response.tool_calls[0]
                tool_name = tool_call.get("name")
                tool_call_id = tool_call.get("id")
                tool_args = tool_call.get("args", {})
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)
            except Exception as e:
                self.logger.warning(f"Error parsing tool_calls: {e}")
                tool_name = None
                tool_args = None
                tool_call_id = None

        elif hasattr(response, "additional_kwargs") and response.additional_kwargs.get(
            "function_call"
        ):
            try:
                fc = response.additional_kwargs["function_call"]
                tool_name = fc.get("name")
                tool_args = json.loads(fc.get("arguments", "{}"))
                tool_call_id = tool_name
            except Exception as e:
                self.logger.warning(f"Error parsing function_call: {e}")
                tool_name = None
                tool_args = None
                tool_call_id = None

        return tool_name, tool_args, tool_call_id

    def _find_tool_func(self, tool_name, tools_to_bind):
        if tool_name:
            for t in tools_to_bind:
                if hasattr(t, "name") and t.name == tool_name:
                    return t

        return AVAILABLE_TOOLS.get(tool_name)

    def _merge_tool_args_with_config(self, tool_args, config_defaults):
        merged_args = dict(tool_args)
        for k, v in config_defaults.items():
            merged_args[k] = v
        return merged_args

    async def _execute_tool_function(
        self, tool_func, tool_name, tool_args, agent_config
    ):
        if tool_func and callable(tool_func):
            try:
                input_fields = []
                if tool_name and tool_name in agent_config.tools:
                    input_fields = agent_config.tools[tool_name].get("input_fields", [])

                if tool_name == "get_position_tool" and input_fields:
                    actual_tool = tool_func(input_fields)
                    tool_func = actual_tool
                    self.logger.debug(
                        f"[GraphService] Created position tool with input_fields: {input_fields}"
                    )

                filtered_args = {}
                if input_fields:

                    config_keys = {
                        "type",
                        "job_type",
                        "input_fields",
                        "description",
                        "required",
                        "variable_name",
                    }
                    filtered_args = {
                        k: v for k, v in tool_args.items() if k not in config_keys
                    }
                else:
                    config_keys = {
                        "type",
                        "job_type",
                        "input_fields",
                        "description",
                        "required",
                        "variable_name",
                    }
                    filtered_args = {
                        k: v for k, v in tool_args.items() if k not in config_keys
                    }

                self.logger.debug(
                    f"[GraphService] Executing tool '{tool_name}' with filtered_args: {filtered_args}"
                )

                tool_result = await tool_func.ainvoke(filtered_args)

                if isinstance(tool_result, ToolMessage):
                    return tool_result
                elif isinstance(tool_result, AIMessage):
                    return ToolMessage(
                        content=tool_result.content, tool_call_id="temp_id"
                    )
                else:
                    return ToolMessage(content=str(tool_result), tool_call_id="temp_id")

            except Exception as ex:
                self.logger.error(
                    f"[GraphService] Error executing tool '{tool_name}': {str(ex)}"
                )
                return ToolMessage(
                    content=f"Error executing tool '{tool_name}': {str(ex)}",
                    tool_call_id="temp_id",
                )

    async def _execute_required_tools(self, agent_config):
        """Execute all required tools for an agent automatically."""

        required_tool_results = []

        if not (
            hasattr(agent_config, "tools") and isinstance(agent_config.tools, dict)
        ):
            return required_tool_results

        for tool_name, tool_config in agent_config.tools.items():
            if not tool_config.get("required", False):
                continue

            self.logger.debug(
                f"[GraphService] Processing required tool '{tool_name}' for agent"
            )

            tool_func = AVAILABLE_TOOLS.get(tool_name)

            if not tool_func:
                self.logger.warning(
                    f"[GraphService] Required tool '{tool_name}' not found in AVAILABLE_TOOLS."
                )
                continue

            try:
                tool_args = dict(tool_config)
                tool_result = await self._execute_tool_function(
                    tool_func, tool_name, tool_args, agent_config
                )
                if tool_result:
                    var_name = tool_config.get("variable_name")
                    required_tool_results.append(
                        {
                            "tool_name": tool_name,
                            "message": tool_result,
                            "variable_name": var_name,
                        }
                    )
                    self.logger.debug(
                        f"[GraphService] Required tool '{tool_name}' executed successfully (context only, not added to messages). Result content: {str(getattr(tool_result, 'content', tool_result))[:200]}..."
                    )
            except Exception as ex:
                self.logger.error(
                    f"[GraphService] Failed to execute required tool '{tool_name}': {str(ex)}"
                )

        return required_tool_results

    def _update_context_with_tool_results(
        self, prompt_context, required_tools_executed, application_attributes=None
    ):
        """Update prompt context with data from required tools results."""
        for result in required_tools_executed:
            tool_name = result.get("tool_name")
            tool_message = result.get("message")
            var_name = result.get("variable_name")

            if not tool_message:
                continue

            content = getattr(tool_message, "content", "")

            self._update_prompt_context(
                prompt_context, tool_name, var_name, content, tool_message
            )

            if application_attributes is not None and content:
                self._update_application_attributes_from_tool(
                    application_attributes, tool_name, content
                )

    def _update_prompt_context(
        self, prompt_context, tool_name, var_name, content, tool_message
    ):
        """Update prompt context with tool result."""
        if var_name:
            prompt_context[var_name] = content or str(tool_message)
        elif tool_name == "get_labels_tool":
            prompt_context["labels"] = content or str(tool_message)

    def _update_application_attributes_from_tool(
        self, application_attributes, tool_name, content
    ):
        """Update application attributes based on tool results."""
        try:
            if content.strip().startswith("{") or content.strip().startswith("["):
                parsed_content = json.loads(content)
                self._process_structured_tool_data(
                    application_attributes, tool_name, parsed_content
                )
            else:
                self._process_text_tool_data(application_attributes, tool_name, content)

            self.logger.debug(
                f"[GraphService] Updated application_attributes from required tool '{tool_name}'"
            )

        except (json.JSONDecodeError, TypeError) as ex:
            self.logger.debug(
                f"[GraphService] Could not parse tool result as JSON for '{tool_name}': {ex}"
            )
            self._process_text_tool_data(application_attributes, tool_name, content)

    def _process_structured_tool_data(
        self, application_attributes, tool_name, parsed_content
    ):
        """Process structured JSON data from tools."""
        if tool_name == "get_positions_tool" and isinstance(parsed_content, list):
            self._extract_position_info_from_list(
                application_attributes, parsed_content
            )
        elif tool_name == "get_labels_tool" and isinstance(parsed_content, dict):
            self._extract_location_info_from_labels(
                application_attributes, parsed_content
            )

    def _extract_position_info_from_list(self, application_attributes, positions):
        """Extract position information from positions data."""
        for position in positions:
            if isinstance(position, dict):
                if "id" in position and not application_attributes.get("position_id"):
                    application_attributes["position_id"] = str(position["id"])
                if "title" in position and not application_attributes.get(
                    "position_name"
                ):
                    application_attributes["position_name"] = str(position["title"])
                break

    def _extract_location_info_from_labels(self, application_attributes, labels_data):
        """Extract location information from labels."""
        if "counties" in labels_data:
            counties = labels_data["counties"]
            if counties and not application_attributes.get("other_information"):
                application_attributes["other_information"] = (
                    f"Available locations: {', '.join(counties[:5])}"
                )

    def _process_text_tool_data(self, application_attributes, tool_name, content):
        """Process text content from tools."""
        if tool_name == "get_positions_tool" and "position" in content.lower():
            self._extract_position_info_from_text(application_attributes, content)

    def _extract_position_info_from_text(self, application_attributes, content):
        """Extract position info from text content."""
        lines = content.split("\n")
        for line in lines[:3]:
            if "id:" in line.lower() and not application_attributes.get("position_id"):
                try:
                    pos_id = line.split(":")[-1].strip()
                    application_attributes["position_id"] = pos_id
                except (IndexError, AttributeError):
                    pass
            elif "title:" in line.lower() and not application_attributes.get(
                "position_name"
            ):
                try:
                    pos_title = line.split(":")[-1].strip()
                    application_attributes["position_name"] = pos_title
                except (IndexError, AttributeError):
                    pass

    def _build_supervisor_prompt(
        self, available_options: list[str], last_agent=None
    ) -> str:
        """Build the system prompt for the supervisor."""

        agent_descriptions = []
        for name, agent in self.graph_config.agents.items():
            if agent.enabled:
                description = (
                    getattr(agent.chain, "description", "")
                    or "No description available"
                )
                agent_descriptions.append(f"- {name}: {description}")

        system_prompt_ending = ""
        if self.graph_config.allow_supervisor_finish:
            system_prompt_ending = "Based on the user input and conversation history, decide which agent should handle this next or if the task is complete (FINISH)."
        else:
            system_prompt_ending = "Based on the user input and conversation history, decide which agent should handle this next. You must select one of the available agents."

        if (
            "recruiter_agent" in self.graph_config.agents.keys()
            or "recruiter" in self.graph_config.agents.keys()
        ):
            system_prompt_ending += "If the user input is smalltalk or asking about the positions. Always choose recruiter agent."

        last_agent_str = (
            f"\nThe last agent selected was: {last_agent}."
            if last_agent
            else "No last agent selected."
        )

        return f"""You are a supervisor managing a team of AI agents. Your job is to decide which agent should handle the user's request{'' if not self.graph_config.allow_supervisor_finish else ' or if the task is complete.'}.

Available agents and their capabilities:
{chr(10).join(agent_descriptions)}

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

        if hasattr(response, "tool_calls") and response.tool_calls:
            try:
                tool_call = response.tool_calls[0]
                if hasattr(tool_call, "function") and hasattr(
                    tool_call.function, "arguments"
                ):
                    args = json.loads(tool_call.function.arguments)
                    return args.get("chain", None)
                elif isinstance(tool_call, dict):
                    return tool_call.get("args", {}).get("chain", None)
            except Exception:
                pass

        if hasattr(response, "additional_kwargs") and response.additional_kwargs.get(
            "function_call"
        ):
            try:
                call = response.additional_kwargs["function_call"]
                args = json.loads(call.get("arguments", "{}"))
                return args.get("chain", None)
            except Exception:
                pass

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

                tool_def = {"type": "function", "function": function_def}

                chain = prompt | llm.bind_tools(
                    tools=[tool_def], tool_choice="required"
                )

                # Filter out ToolMessage objects to avoid OpenAI API errors
                filtered_messages = self._filter_problematic_messages(state.messages)
                response = await chain.ainvoke({"messages": filtered_messages})

                next_agent = self._extract_next_agent_from_response(response)

                if next_agent not in available_options:
                    if self.graph_config.allow_supervisor_finish:
                        next_agent = "FINISH"
                    else:
                        next_agent = enabled_agents[0]

                state.next = next_agent

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
                state.last_agent = agent_name

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

    def _extract_tool_call_ids(self, tool_calls):
        """Extract tool call IDs from tool_calls."""
        tool_call_ids = set()
        for tool_call in tool_calls:
            if hasattr(tool_call, "id"):
                tool_call_ids.add(tool_call.id)
            elif isinstance(tool_call, dict) and "id" in tool_call:
                tool_call_ids.add(tool_call["id"])
        return tool_call_ids

    def _collect_tool_messages(self, messages, start_index, expected_tool_call_ids):
        """Collect ToolMessages that match expected tool_call_ids."""
        j = start_index
        found_tool_messages = []
        found_tool_call_ids = set()

        while j < len(messages) and isinstance(messages[j], ToolMessage):
            tool_msg = messages[j]
            if (
                hasattr(tool_msg, "tool_call_id")
                and tool_msg.tool_call_id in expected_tool_call_ids
            ):
                found_tool_messages.append(tool_msg)
                found_tool_call_ids.add(tool_msg.tool_call_id)
            j += 1

        return found_tool_messages, found_tool_call_ids, j

    def _process_ai_message_with_tools(self, msg, messages, i, filtered):
        """Process AIMessage that has tool_calls."""
        tool_call_ids = self._extract_tool_call_ids(msg.tool_calls)

        # Collect following ToolMessages
        found_tool_messages, found_tool_call_ids, next_index = (
            self._collect_tool_messages(messages, i + 1, tool_call_ids)
        )

        # If we have unmatched tool_calls, convert this AIMessage to a regular message
        if tool_call_ids and tool_call_ids != found_tool_call_ids:
            content = msg.content if msg.content else "Tool execution completed."
            filtered.append(AIMessage(content=content))
        else:
            # Keep the AIMessage and its corresponding ToolMessages
            filtered.append(msg)
            filtered.extend(found_tool_messages)

        return next_index

    def _filter_problematic_messages(self, messages):
        """Filter out problematic message sequences that cause OpenAI API errors."""
        if not messages:
            return []

        filtered = []
        i = 0

        while i < len(messages):
            msg = messages[i]

            # If this is an AIMessage with tool_calls
            if (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                i = self._process_ai_message_with_tools(msg, messages, i, filtered)
            elif isinstance(msg, ToolMessage):
                # Orphaned ToolMessage - convert to AIMessage
                converted_msg = AIMessage(content=f"[Tool Result]: {msg.content}")
                filtered.append(converted_msg)
                i += 1
            else:
                # Regular message
                filtered.append(msg)
                i += 1

        return filtered

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

        workflow.add_node(
            "applicant_attributes_extractor",
            self._applicant_attributes_extractor_node,
        )
        for agent_name in enabled_agents.keys():
            workflow.add_edge(agent_name, "applicant_attributes_extractor")
        workflow.add_edge("applicant_attributes_extractor", END)

        if self.graph_config.exception_chain:
            workflow.add_edge("exception_chain", END)

        conditional_mapping = {
            agent_name: agent_name for agent_name in enabled_agents.keys()
        }

        if self.graph_config.allow_supervisor_finish:
            conditional_mapping["FINISH"] = END

        workflow.add_conditional_edges(
            "supervisor",
            self._should_continue_from_supervisor,
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
                self._should_continue_from_topic_validator,
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

        self.logger.debug("[GraphService] Graph execution completed successfully.")
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
            response = await chain.ainvoke(
                {"messages": [HumanMessage(content=user_input)]}
            )

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
            user_input = self._get_user_input_from_state(state)
            if not user_input:
                self.logger.warning(
                    f"[GraphService] No user input found for topic validation. State: {state}"
                )
                return state

            topic_validator_service = TopicValidatorService()

            allowed_topics = getattr(topic_validation_config, "allowed_topics", None)
            invalid_topics = getattr(topic_validation_config, "invalid_topics", None)
            config: Model = getattr(topic_validation_config, "model", None)

            if not config:
                raise ValueError("Topic validation model configuration is required")

            is_valid, topic, reason = await topic_validator_service.validate_topic(
                question=user_input,
                provider=config.provider.value,
                name=config.name,
                deployment=config.deployment,
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

            state["next"] = "supervisor"
            return state

        except Exception as ex:
            self.logger.error(
                f"[GraphService] Error in topic validator: {str(ex)}. State: {state}"
            )
            if self.graph_config.exception_chain:
                state["next"] = "exception_chain"
                return state
            return state

    async def _exception_chain_node(self, state: AgentState) -> AgentState:
        """Exception chain node for handling errors and invalid inputs."""
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
            user_input = self._get_user_input_from_state(state)
            context = f"User input: {user_input}"
            prompt = prompt.partial(context=context)
        except Exception as e:
            self.logger.error(
                f"Failed to load exception chain prompt {self.graph_config.exception_chain.chain.prompt_id} from {self.tracer_type}: {str(e)}"
            )
            return self._handle_exception_chain_fallback(state)

        try:
            chain = prompt | llm

            # Filter out ToolMessage objects to avoid OpenAI API errors
            filtered_messages = self._filter_problematic_messages(state["messages"])
            response = await chain.ainvoke({"messages": filtered_messages})
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
        """Handle fallback when exception chain fails."""
        error_message = AIMessage(
            content="I apologize, but I'm unable to process your request at this time. Please try again later."
        )
        state["messages"] = add_messages(state["messages"], [error_message])
        state["next"] = "FINISH"
        return state

    def _should_continue_from_supervisor(
        self, state: AgentState
    ) -> Literal["FINISH"] | str:
        """Determine next step after supervisor node."""

        if state["next"] == "FINISH":
            return "FINISH"
        return state["next"]

    def _should_continue_from_topic_validator(
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

    def _get_user_input_from_state(self, state: AgentState) -> str:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage) and hasattr(msg, "content"):
                return msg.content
        return ""


# For LangGraph Studio integration
graph_service = GraphService(AppSettingsService())


async def get_compiled_workflow_for_studio(config: RunnableConfig):
    app_id = config.get("app_id")
    parameters = config.get("parameters")
    return await graph_service.get_compiled_workflow(app_id, parameters)
