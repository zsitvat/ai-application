import asyncio
import json
import os
from typing import Any, Literal

import aiohttp
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from functools import partial

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.schemas.graph_schema import (
    Agent,
    AgentState,
    ApplicationAttributes,
    ExtractorConfig,
    GraphConfig,
    Model,
)
from src.services.graph.graph_checkpointer import create_checkpointer
from src.services.graph.tool_handler import ToolHandler
from src.services.validators.topic_validator.topic_validator_service import (
    TopicValidatorService,
)
from src.services.validators.personal_data.personal_data_filter_service import (
    PersonalDataFilterService,
)
from src.utils.extract_message_content import extract_message_content
from src.utils.get_prompt import get_prompt_by_type
from src.utils.select_model import get_chat_model


class Graph:
    """Handles the construction and execution of the agent workflow graph."""

    def __init__(
        self,
        logger: Any,
        app_settings_service: Any,
        graph_config: GraphConfig | None = None,
        tracer_type: str = None,
    ):
        self.graph_config = graph_config
        self.logger = logger
        self.app_settings_service = app_settings_service
        self.workflow = None
        self.tracer_type = tracer_type

        # Initialize tool handler
        self.tool_handler = ToolHandler()

    async def get_compiled_workflow(
        self, app_id: int, parameters: dict[str, Any] | None = None
    ) -> StateGraph:
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
            raise ex

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
            raise ex

    async def prepare_graph_execution(
        self, graph_config: dict | GraphConfig | None, user_input: str
    ) -> str:
        """
        Prepare graph for execution by setting config and validating input.
        If the input exceeds max length, it will be truncated.
        """
        self.graph_config = graph_config
        if not self.graph_config or not self.graph_config.get("agents"):
            raise ValueError("No valid graph configuration or agents found")

        if isinstance(self.graph_config, dict):
            self.graph_config = GraphConfig(**self.graph_config)
        if (
            getattr(self.graph_config, "max_input_length", 0) > 0
            and len(user_input) > self.graph_config.max_input_length
        ):
            user_input = user_input[: self.graph_config.max_input_length]
            self.logger.debug(
                f"[GraphService] User input truncated to {self.graph_config.max_input_length} characters"
            )
        if not hasattr(self, "workflow") or self.workflow is None:
            workflow_builder = self._build_workflow()
            checkpointer = await self._create_checkpointer()
            self.workflow = workflow_builder.compile(checkpointer=checkpointer)
        return user_input

    async def _applicant_attributes_extractor_node(
        self, state: AgentState
    ) -> AgentState:
        """Node that extracts applicant attributes from all messages using LLM and updates application_attributes."""

        try:
            extractor_config: ExtractorConfig = getattr(
                self.graph_config, "applicant_attributes_extractor"
            )
            if extractor_config is None:
                return state

            llm = get_chat_model(
                provider=extractor_config.model.provider.value,
                deployment=extractor_config.model.deployment,
                model=extractor_config.model.name,
            ).with_structured_output(schema=ApplicationAttributes)

            prompt = await get_prompt_by_type(
                prompt_id=extractor_config.prompt_id, tracer_type=self.tracer_type
            )

            messages_str = "\n".join(
                [extract_message_content(m) for m in state.messages]
            )

            extractor_context = dict(state.context) if state.context else {}

            for var in getattr(prompt, "input_variables", []):
                if var != "messages":
                    extractor_context[var] = ""

            response = await (prompt | llm).ainvoke(
                {"messages": messages_str, **extractor_context}
            )

            if isinstance(response, dict):
                for key, value in response.items():
                    if key in state.application_attributes and value:
                        state.application_attributes[key] = str(value)

                # Check if all required fields are present and call endpoint if so
                if self._check_required_fields_complete(state.application_attributes):
                    await self._submit_application_data(state.application_attributes)

                    state.application_attributes = dict.fromkeys(
                        state.application_attributes, ""
                    )

                    state.messages.append(
                        AIMessage(
                            content="A jelentkező adat(ok) mentése sikeresen megtörtént.",
                        )
                    )

            return state

        except Exception as ex:
            self.logger.error(
                f"[GraphService|_applicant_attributes_extractor_node] error: {str(ex)}"
            )
            return state

    async def _personal_data_filter_node(self, state: AgentState) -> AgentState:
        try:
            personal_data_filter_config = self._find_personal_data_filter_config()
            if not personal_data_filter_config:
                self.logger.info("[GraphService] Personal data filter is disabled")
                return state

            filter_service = PersonalDataFilterService()

            if state.application_attributes:
                filtered_attributes = {}

                for key, value in state.application_attributes.items():
                    if isinstance(value, str) and value.strip():
                        try:
                            filtered_value = await filter_service.filter_personal_data(
                                text=value,
                                model=personal_data_filter_config.chain.model,
                                sensitive_words=getattr(
                                    personal_data_filter_config,
                                    "sensitive_data_types",
                                    [],
                                ),
                                regex_patterns=self._get_default_regex_patterns(),
                                prompt=personal_data_filter_config.chain.prompt_id,
                                mask_char=getattr(
                                    personal_data_filter_config, "mask_char", "*"
                                ),
                            )
                            filtered_attributes[key] = filtered_value
                            self.logger.info(
                                f"[GraphService] Filtered attribute: {key}"
                            )
                        except Exception as filter_ex:
                            self.logger.error(
                                f"[GraphService] Error filtering {key}: {str(filter_ex)}"
                            )
                            filtered_attributes[key] = value
                    else:
                        filtered_attributes[key] = value

                state.application_attributes = filtered_attributes
                self.logger.info(
                    "[GraphService] Personal data filtering completed for application attributes"
                )

            return state

        except Exception as ex:
            self.logger.error(
                f"[GraphService|_personal_data_filter_node] error: {str(ex)}"
            )
            return state

    def _get_default_regex_patterns(self) -> list[str]:
        return [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            r"\+?36[- ]?\d{1,2}[- ]?\d{3}[- ]?\d{4}",
            r"\b\d{3}-\d{2}-\d{4}\b",
        ]

    async def _create_checkpointer(self):

        redis_url = os.getenv("REDIS_URL")
        find_personal_data_filter_config = getattr(
            self, "_find_personal_data_filter_config", lambda: None
        )
        return await create_checkpointer(
            graph_config=self.graph_config,
            logger=self.logger,
            redis_url=redis_url,
            find_personal_data_filter_config=find_personal_data_filter_config,
        )

    def _check_required_fields_complete(
        self, application_attributes: dict[str, Any]
    ) -> bool:
        """Check if all required application attributes are present and not empty."""

        required_fields = [
            "applicant_name",
            "phone_number",
            "position_name",
            "position_id",
            "application_reason",
            "experience",
            "email",
            "cv",
        ]

        required_fields = getattr(
            self.graph_config.applicant_attributes_extractor,
            "required_attributes",
            required_fields,
        )

        for field in required_fields:
            value = application_attributes.get(field, "").strip()
            if not value:
                return False

        self.logger.debug("[GraphService] All required fields are complete")
        return True

    async def _submit_application_data(
        self, application_attributes: dict[str, Any]
    ) -> None:
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
            prompt = self.tool_handler.inject_tool_info_into_prompt(
                prompt, agent_config
            )
            prompt_with_messages = ChatPromptTemplate.from_messages(
                [
                    prompt,
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            tools_to_bind = self.tool_handler.bind_tools_to_chain(agent_config)
            chain = prompt_with_messages | llm.bind_tools(
                tools_to_bind, tool_choice=agent_config.tool_choice
            )
            prompt_context = self._build_prompt_context(state, agent_config)
            self._update_state_context(state, prompt_context)
            required_tools_executed = await self.tool_handler.execute_required_tools(
                agent_config
            )
            self.tool_handler.update_context_with_tool_results(
                prompt_context,
                required_tools_executed,
                state.application_attributes,
                state=state,
            )
            self.logger.debug(
                f"[GraphService] Agent '{agent_name}' application_attributes after required tools: {state.application_attributes}"
            )
            self.logger.debug(
                f"[GraphService] Agent '{agent_name}' prompt_context keys after required tools: {list(prompt_context.keys())}"
            )
            response = await chain.ainvoke(prompt_context)
            tool_name, tool_args, tool_call_id = self.tool_handler.extract_tool_call(
                response
            )
            tool_func = self.tool_handler.find_tool_func(tool_name, tools_to_bind)
            state["messages"] = add_messages(state["messages"], [response])
            if tool_name and tool_func:
                self.logger.debug(
                    f"[GraphService] Original tool args from LLM for '{tool_name}': {tool_args}"
                )
                tool_args = self.tool_handler.prepare_tool_args(
                    tool_name, tool_args, agent_config
                )
                ran_tool = await self.tool_handler.run_tool_and_update_state(
                    state,
                    tool_func,
                    tool_name,
                    tool_args,
                    agent_config,
                    tool_call_id,
                    prompt_context,
                    self.tool_handler.update_context_with_tool_results,
                    self._filter_problematic_messages,
                )
                if ran_tool:
                    follow_up_response = await (prompt_with_messages | llm).ainvoke(
                        prompt_context
                    )
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

    def _build_supervisor_prompt(
        self, available_options: list[str], last_agent: str | None = None
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
            system_prompt_ending += "If the user input is smalltalk or asking about the jobs. Always choose recruiter agent."

        last_agent_str = (
            f"\nThe last agent selected was: {last_agent}."
            if last_agent
            else "No last agent selected."
        )

        supervisor_description_text = (
            self.graph_config.supervisor.chain.description or ""
        )

        return f"""You are a supervisor managing a team of AI agents. Your job is to decide which agent should handle the user's request{'' if not self.graph_config.allow_supervisor_finish else ' or if the task is complete.'}.

Available agents and their capabilities:
{chr(10).join(agent_descriptions)}

{supervisor_description_text}
{last_agent_str}s

{system_prompt_ending}s

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

    def _extract_next_agent_from_response(self, response: Any) -> str | None:
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
                self.logger.warning(
                    f"[GraphService] Error extracting next agent from tool_calls: {str(response.tool_calls)}"
                )
                return None

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

    def _create_supervisor_node(self) -> Any:
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

                state.next_agent = next_agent

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
                    f"[GraphService] Supervisor node decided next action: {state['next_agent']}. State: {state}"
                )
                return state

            except Exception as ex:
                self.logger.error(
                    f"[GraphService] Error in supervisor node: {str(ex)}. State: {state}"
                )
                raise

        return supervisor_node

    def _extract_tool_call_ids(self, tool_calls: list[Any]) -> set:
        """Extract tool call IDs from tool_calls."""
        tool_call_ids = set()
        for tool_call in tool_calls:
            if hasattr(tool_call, "id"):
                tool_call_ids.add(tool_call.id)
            elif isinstance(tool_call, dict) and "id" in tool_call:
                tool_call_ids.add(tool_call["id"])
        return tool_call_ids

    def _collect_tool_messages(
        self, messages: list[Any], start_index: int, expected_tool_call_ids: set
    ) -> tuple[list, set, int]:
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

    def _process_ai_message_with_tools(
        self, msg: AIMessage, messages: list[Any], i: int, filtered: list[Any]
    ) -> int:
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

    def _filter_problematic_messages(self, messages: list[Any]) -> list[Any]:
        """Filter out problematic message sequences that cause OpenAI API errors."""
        if not messages:
            return []

        filtered = []
        i = 0

        while i < len(messages):
            msg = messages[i]

            if (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                i = self._process_ai_message_with_tools(msg, messages, i, filtered)
            elif isinstance(msg, ToolMessage):
                converted_msg = AIMessage(content=f"[Tool Result]: {msg.content}")
                filtered.append(converted_msg)
                i += 1
            else:
                filtered.append(msg)
                i += 1

        return filtered

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with supervisor pattern."""

        workflow = StateGraph(AgentState)

        if self._is_topic_validator_enabled():
            workflow.add_node("topic_validator", self._topic_validator_node)

        workflow.add_node("supervisor", self._create_supervisor_node())
        workflow.add_node(
            "applicant_attributes_extractor", self._applicant_attributes_extractor_node
        )
        workflow.add_node("personal_data_filter", self._personal_data_filter_node)
        workflow.add_node("exception_chain", self._exception_chain_node)

        enabled_agents = self._get_enabled_agents()
        for agent_name, agent_config in enabled_agents.items():
            agent_node = partial(
                self._agent_node, agent_name=agent_name, agent_config=agent_config
            )
            workflow.add_node(agent_name, agent_node)

        for agent_name in enabled_agents.keys():
            workflow.add_edge(agent_name, "applicant_attributes_extractor")
        workflow.add_edge("applicant_attributes_extractor", "personal_data_filter")
        workflow.add_edge("personal_data_filter", END)

        self._configure_conditional_edges(workflow, enabled_agents)

        entry_point = (
            "topic_validator" if self._is_topic_validator_enabled() else "supervisor"
        )
        workflow.set_entry_point(entry_point)

        return workflow

    def _get_enabled_agents(self) -> dict:
        """
        Returns the enabled agents as a dict of name: agent object pairs,
        where agent.enabled is True and the object is of type Agent.
        """
        if not self.graph_config or not hasattr(self.graph_config, "agents"):
            return {}
        return {
            name: agent
            for name, agent in self.graph_config.agents.items()
            if getattr(agent, "enabled", False) and isinstance(agent, Agent)
        }

    def _is_topic_validator_enabled(self) -> bool:
        """
        Checks if the topic_validator is enabled in the graph_config.
        """
        topic_validator = getattr(self.graph_config, "topic_validator", None)
        return bool(topic_validator and getattr(topic_validator, "enabled", False))

    def _configure_conditional_edges(
        self, workflow: StateGraph, enabled_agents: dict
    ) -> None:
        """
        Configure conditional edges for the workflow graph, including supervisor and topic_validator nodes.
        """
        # Supervisor conditional mapping
        conditional_mapping = {
            agent_name: agent_name for agent_name in enabled_agents.keys()
        }
        if getattr(self.graph_config, "allow_supervisor_finish", False):
            conditional_mapping["FINISH"] = "applicant_attributes_extractor"
        workflow.add_conditional_edges(
            "supervisor",
            self._should_continue_from_supervisor,
            conditional_mapping,
        )

        # Topic validator conditional mapping (if topic validator is enabled)
        if self._is_topic_validator_enabled():
            topic_conditional_mapping = {
                "supervisor": "supervisor",
                "exception_chain": "exception_chain",
                "FINISH": END,
            }
            workflow.add_conditional_edges(
                "topic_validator",
                self._should_continue_from_topic_validator,
                topic_conditional_mapping,
            )

    def _generate_final_response(self, result: dict) -> str:
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

    def _find_topic_validation_config(self) -> Any:
        """Return topic validator config if enabled, else None."""

        config = getattr(self.graph_config, "topic_validator", None)
        if config and getattr(config, "enabled", False):
            return config
        return None

    def _handle_invalid_topic(self, state: AgentState, reason: str) -> AgentState:
        """Handle when topic validation fails."""

        if self.graph_config.exception_chain:
            state["next_agent"] = "exception_chain"
            self.logger.debug(
                f"[GraphService] Question rejected by topic validator, routing to exception chain. Reason: {reason}, State: {state}"
            )
        else:
            error_message = AIMessage(
                content="I can only help with work-related topics. Please ask questions about careers, job search, professional development, or workplace matters."
            )
            state["messages"] = add_messages(state["messages"], [error_message])
            state["next_agent"] = "FINISH"
            self.logger.debug(
                f"[GraphService] Question rejected by topic validator, no exception chain available. Reason: {reason}, State: {state}"
            )
        return state

    def _find_personal_data_filter_config(self) -> Any:
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

            state["next_agent"] = "supervisor"
            return state

        except Exception as ex:
            self.logger.error(
                f"[GraphService] Error in topic validator: {str(ex)}. State: {state}"
            )
            if self.graph_config.exception_chain:
                state["next_agent"] = "exception_chain"
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
            state["next_agent"] = "FINISH"
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
        state["next_agent"] = "FINISH"
        return state

    def _should_continue_from_supervisor(
        self, state: AgentState
    ) -> Literal["FINISH"] | str:
        """Determine next step after supervisor node."""

        if state["next_agent"] == "FINISH":
            return "FINISH"
        return state["next_agent"]

    def _should_continue_from_topic_validator(
        self, state: AgentState
    ) -> Literal["supervisor", "exception_chain", "FINISH"]:
        """Determine next step after topic validator node."""

        next_step = state.get("next_agent")
        if next_step == "exception_chain":
            return "exception_chain"
        elif next_step == "FINISH":
            return "FINISH"
        else:
            return "supervisor"

    def _get_user_input_from_state(self, state: AgentState) -> str:
        """Extract user input from the agent state."""

        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage) and hasattr(msg, "content"):
                if isinstance(msg.content, str):
                    return msg.content
                else:
                    return str(msg.content)
        return ""

    def _build_prompt_context(self, state: AgentState, agent_config: Agent) -> dict:
        """Build the prompt context for the given state and agent configuration."""

        prompt_context = {
            "messages": state.messages,
            "application_attributes": getattr(state, "application_attributes", {}),
            "user_information": [],
            "labels": [],
            "positions": [],
        }

        if hasattr(state, "parameters") and state.parameters:
            app_id_val = state.parameters.get("app_id")
            if app_id_val is not None:
                prompt_context["app_id"] = app_id_val
            prompt_context.update(state.parameters)

        if hasattr(agent_config, "tools") and isinstance(agent_config.tools, dict):
            for tool_name, tool_config in agent_config.tools.items():
                if isinstance(tool_config, dict):
                    prompt_context.update(
                        {
                            k: v
                            for k, v in tool_config.items()
                            if k not in prompt_context
                        }
                    )

        if state.context:
            prompt_context.update(state.context)
        return prompt_context

    def _update_state_context(self, state: AgentState, prompt_context: dict) -> None:
        """Update the state context with prompt context attributes."""
        state.context["application_attributes"] = prompt_context[
            "application_attributes"
        ]
        for k, v in prompt_context.items():
            if k != "messages":
                state.context[k] = v
