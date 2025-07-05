import asyncio
import importlib
import json
import logging
import os
from functools import partial
from typing import Any, Literal
from urllib.parse import urlparse

import redis
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from schemas.graph_schema import (
    Agent,
    AgentState,
    Chain,
    CheckpointerType,
    GraphConfig,
    Model,
    ModelProviderType,
    ModelType,
)
from services.data_api.app_settings import AppSettingsService
from utils.select_model import get_chat_model

# Constants
NO_AGENTS_ERROR = "No enabled agents available and supervisor finish is disabled"


class GraphService:
    """Service for handling multi-agent graph execution with supervisor pattern."""

    def __init__(self, app_settings_service: AppSettingsService):
        self.logger = logging.getLogger(__name__)
        self.app_settings_service = app_settings_service
        self.graph_config: GraphConfig | None = None
        self.workflow = None

        self.redis_url = (
            f"redis://{os.getenv('REDIS_USER')}:{os.getenv('REDIS_PASSWORD')}@{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}"
            if os.getenv("REDIS_PASSWORD")
            else f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}"
        )

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

            self.logger.info(
                f"Loaded graph config with {len(self.graph_config.agents)} agents"
            )

        except Exception as e:
            self.logger.error(f"Error loading graph configuration: {str(e)}")
            raise

    async def _agent_node(
        self, state: AgentState, agent_name: str, agent_config: Agent
    ) -> AgentState:
        """Handle agent node execution."""
        try:
            llm = await get_chat_model(
                provider=agent_config.chain.model.provider.value,
                deployment=agent_config.chain.model.deployment,
                model=agent_config.chain.model.name,
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=agent_config.chain.prompt),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )

            chain = prompt | llm
            response = await chain.ainvoke({"messages": state["messages"]})
            state["messages"] = add_messages(state["messages"], [response])

            self.logger.info(f"Agent {agent_name} completed processing")
            return state

        except Exception as e:
            self.logger.error(f"Error in agent {agent_name}: {str(e)}")

            error_message = AIMessage(
                content=f"Agent {agent_name} encountered an error: {str(e)}"
            )
            state["messages"] = add_messages(state["messages"], [error_message])
            return state

    def _create_supervisor_node(self):
        """Create the supervisor node that decides which agent to call next."""

        async def supervisor_node(state: AgentState) -> AgentState:
            try:
                llm = await get_chat_model(
                    provider=self.graph_config.supervisor.chain.model.provider.value,
                    deployment=self.graph_config.supervisor.chain.model.deployment,
                    model=self.graph_config.supervisor.chain.model.name,
                )

                enabled_agents = [
                    name
                    for name, agent in self.graph_config.agents.items()
                    if agent.enabled
                ]

                if not enabled_agents:
                    raise ValueError(NO_AGENTS_ERROR)

                available_options = enabled_agents[:]
                if self.graph_config.allow_supervisor_finish:
                    available_options.append("FINISH")

                function_def = {
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

                agent_descriptions = []
                for name, agent in self.graph_config.agents.items():
                    if agent.enabled:
                        agent_descriptions.append(
                            f"- {name}: {agent.chain.prompt[:100]}..."
                        )

                system_prompt_ending = ""
                if self.graph_config.allow_supervisor_finish:
                    system_prompt_ending = "Based on the user input and conversation history, decide which agent should handle this next or if the task is complete (FINISH)."
                else:
                    system_prompt_ending = "Based on the user input and conversation history, decide which agent should handle this next. You must select one of the available agents."

                system_prompt = f"""You are a supervisor managing a team of AI agents. Your job is to decide which agent should handle the user's request{"" if not self.graph_config.allow_supervisor_finish else " or if the task is complete"}.

Available agents and their capabilities:
{chr(10).join(agent_descriptions)}

{system_prompt_ending}

Select one of: {available_options}"""

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

                if response.additional_kwargs.get("function_call"):
                    next_agent = json.loads(
                        response.additional_kwargs["function_call"]["arguments"]
                    ).get("chain", None)
                elif response.tool_calls:
                    next_agent = (
                        response.tool_calls[0].get("args", {}).get("chain", None)
                    )
                else:
                    next_agent = None

                if next_agent not in available_options:
                    if self.graph_config.allow_supervisor_finish:
                        next_agent = "FINISH"
                    else:
                        next_agent = enabled_agents[0]

                state["next"] = next_agent

                self.logger.info(f"Supervisor decided next action: {state['next']}")
                return state

            except Exception as e:
                self.logger.error(f"Error in supervisor: {str(e)}")
                raise

        return supervisor_node

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with supervisor pattern."""
        workflow = StateGraph(AgentState)

        supervisor_node = self._create_supervisor_node()
        workflow.add_node("supervisor", supervisor_node)

        enabled_agents = {
            name: agent
            for name, agent in self.graph_config.agents.items()
            if agent.enabled
        }

        if not enabled_agents and not self.graph_config.allow_supervisor_finish:
            raise ValueError(NO_AGENTS_ERROR)

        for agent_name, agent_config in enabled_agents.items():
            agent_node = partial(
                self._agent_node, agent_name=agent_name, agent_config=agent_config
            )
            workflow.add_node(agent_name, agent_node)

        for agent_name in enabled_agents.keys():
            workflow.add_edge(agent_name, "supervisor")

        def should_continue(state: AgentState) -> Literal["FINISH"] | str:
            if state["next"] == "FINISH":
                return "FINISH"
            return state["next"]

        conditional_mapping = {
            agent_name: agent_name for agent_name in enabled_agents.keys()
        }

        if self.graph_config.allow_supervisor_finish:
            conditional_mapping["FINISH"] = END

        workflow.add_conditional_edges(
            "supervisor",
            should_continue,
            conditional_mapping,
        )

        workflow.set_entry_point("supervisor")

        return workflow

    def _create_checkpointer(self):
        """Create checkpointer based on configuration."""
        checkpointer_type = self.graph_config.checkpointer_type

        if checkpointer_type == CheckpointerType.MEMORY:
            return InMemorySaver()
        elif checkpointer_type == CheckpointerType.REDIS:
            try:
                redis_url = (
                    self.redis_url
                    if os.getenv("REDIS_HOST")
                    else "redis://localhost:6379/0"
                )
                parsed = urlparse(redis_url)

                redis_config = {
                    "host": parsed.hostname or "localhost",
                    "port": parsed.port or 6379,
                    "db": (
                        int(parsed.path.lstrip("/"))
                        if parsed.path and parsed.path != "/"
                        else 0
                    ),
                    "decode_responses": True,
                    "password": parsed.password if parsed.password else None,
                }

                redis_client = redis.Redis(**redis_config)
                return RedisSaver(redis_client)
            except Exception as e:
                self.logger.warning(
                    f"Failed to create Redis checkpointer: {e}, falling back to memory"
                )
                return InMemorySaver()
        elif checkpointer_type == CheckpointerType.CUSTOM:
            try:
                if hasattr(self.graph_config, "custom_checkpointer_class"):
                    module_path, class_name = (
                        self.graph_config.custom_checkpointer_class.rsplit(".", 1)
                    )

                    module = importlib.import_module(module_path)
                    checkpointer_class = getattr(module, class_name)

                    return checkpointer_class()
                else:
                    return RedisChatHistory()
            except ImportError as e:
                self.logger.warning(
                    f"Custom checkpointer class not available: {e}, falling back to memory checkpointer"
                )
                return InMemorySaver()
            except Exception as e:
                self.logger.warning(
                    f"Failed to create custom checkpointer: {e}, falling back to memory"
                )
                return InMemorySaver()
        else:
            self.logger.warning(
                f"Unknown checkpointer type: {checkpointer_type}, falling back to memory"
            )
            return InMemorySaver()

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
        try:
            self.logger.info(f"Executing graph for user_input: {user_input}")

            await self._prepare_graph_execution(app_id, parameters, user_input)

            initial_state = self._create_initial_state(user_input, context, parameters)

            if self.graph_config.enable_checkpointer:
                config = self._create_graph_config(user_id)
                result = await self.workflow.ainvoke(initial_state, config)
            else:
                result = await self.workflow.ainvoke(initial_state)

            return self._generate_final_response(result)

        except Exception as e:
            self.logger.error(f"Error executing graph: {str(e)}")
            return await self._handle_execution_error(user_input, str(e))

    async def _prepare_graph_execution(
        self, app_id: int, parameters: dict | None, user_input: str
    ) -> None:
        """Prepare graph for execution by loading config and validating input."""
        await self._load_graph_configuration(app_id, parameters)

        if not self.graph_config or not self.graph_config.agents:
            raise ValueError("No valid graph configuration or agents found")

        if not self.workflow:
            workflow_builder = self._build_workflow()

            if self.graph_config.enable_checkpointer:
                checkpointer = self._create_checkpointer()
                self.workflow = workflow_builder.compile(checkpointer=checkpointer)
            else:
                self.workflow = workflow_builder.compile()

        if (
            self.graph_config.max_input_length > 0
            and len(user_input) > self.graph_config.max_input_length
        ):
            raise ValueError(
                f"Input length ({len(user_input)}) exceeds maximum allowed ({self.graph_config.max_input_length})"
            )

    def _create_initial_state(
        self, user_input: str, context: dict | None, parameters: dict | None
    ) -> AgentState:
        """Create the initial state for graph execution."""
        return AgentState(
            messages=[HumanMessage(content=user_input)],
            next="",
            user_input=user_input,
            context=context or {},
            parameters=parameters or {},
        )

    def _create_graph_config(self, user_id: str | None) -> dict:
        """Create configuration for graph execution with checkpointing based on user_id."""
        thread_id = user_id or "anonymous"
        return {"configurable": {"thread_id": thread_id}}

    def _generate_final_response(self, result: AgentState) -> str:
        """Generate the final response from agent results."""
        if result["messages"] and len(result["messages"]) > 1:
            last_message = result["messages"][-1]
            if hasattr(last_message, "content"):
                final_response = last_message.content
            else:
                final_response = str(last_message)
        else:
            final_response = "No response generated from agents."

        self.logger.info("Graph execution completed successfully")
        return final_response

    async def _handle_execution_error(self, user_input: str, error_message: str) -> str:
        """Handle execution errors, potentially using exception chain."""
        if (
            hasattr(self, "graph_config")
            and self.graph_config
            and self.graph_config.exception_chain
        ):
            try:
                return await self._handle_exception_with_chain(
                    user_input, error_message
                )
            except Exception as chain_error:
                self.logger.error(f"Exception chain also failed: {str(chain_error)}")

        return f"Error executing multi-agent workflow: {error_message}"

    async def _handle_exception_with_chain(
        self, user_input: str, error_message: str
    ) -> str:
        """Handle exceptions using the configured exception chain."""
        try:
            llm = await get_chat_model(
                provider=self.graph_config.exception_chain.chain.model.provider.value,
                deployment=self.graph_config.exception_chain.chain.model.deployment,
                model=self.graph_config.exception_chain.chain.model.name,
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=f"{self.graph_config.exception_chain.chain.prompt}\n\nUser input: {user_input}\nError: {error_message}"
                    ),
                ]
            )

            chain = prompt | llm
            response = await chain.ainvoke({})

            self.logger.info("Exception handled by exception chain")
            return response.content

        except Exception as e:
            self.logger.error(f"Exception chain failed: {str(e)}")
            raise

    def get_compiled_workflow(self, app_id: int, parameters: dict | None = None):
        """Get the compiled workflow for LangGraph Studio integration."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.run_until_complete(
                    self._load_graph_configuration(app_id, parameters)
                )
            else:
                asyncio.run(self._load_graph_configuration(app_id, parameters))

            if not self.workflow:
                workflow_builder = self._build_workflow()

                if self.graph_config.enable_checkpointer:
                    checkpointer = self._create_checkpointer()
                    self.workflow = workflow_builder.compile(checkpointer=checkpointer)
                else:
                    self.workflow = workflow_builder.compile()

            return self.workflow
        except Exception as e:
            self.logger.error(f"Error getting compiled workflow: {str(e)}")
            raise

    def export_graph_for_studio(
        self, app_id: int, parameters: dict | None = None
    ) -> dict:
        """Export graph configuration for LangGraph Studio."""
        try:
            self.get_compiled_workflow(app_id, parameters)

            graph_info = {
                "nodes": [],
                "edges": [],
                "config": {
                    "checkpointer_enabled": self.graph_config.enable_checkpointer,
                    "checkpointer_type": (
                        self.graph_config.checkpointer_type.value
                        if self.graph_config.checkpointer_type
                        else None
                    ),
                    "max_input_length": self.graph_config.max_input_length,
                    "agents_count": len(self.graph_config.agents),
                    "allow_supervisor_finish": self.graph_config.allow_supervisor_finish,
                },
            }

            enabled_agents = {
                name: agent
                for name, agent in self.graph_config.agents.items()
                if agent.enabled
            }

            graph_info["nodes"].append(
                {
                    "id": "supervisor",
                    "type": "supervisor",
                    "config": {
                        "model": {
                            "provider": self.graph_config.supervisor.chain.model.provider.value,
                            "name": self.graph_config.supervisor.chain.model.name,
                            "deployment": self.graph_config.supervisor.chain.model.deployment,
                        }
                    },
                }
            )

            for agent_name, agent_config in enabled_agents.items():
                graph_info["nodes"].append(
                    {
                        "id": agent_name,
                        "type": "agent",
                        "config": {
                            "model": {
                                "provider": agent_config.chain.model.provider.value,
                                "name": agent_config.chain.model.name,
                                "deployment": agent_config.chain.model.deployment,
                            },
                            "prompt": (
                                agent_config.chain.prompt[:200] + "..."
                                if len(agent_config.chain.prompt) > 200
                                else agent_config.chain.prompt
                            ),
                        },
                    }
                )

                graph_info["edges"].append({"from": agent_name, "to": "supervisor"})

            conditional_targets = list(enabled_agents.keys())
            if self.graph_config.allow_supervisor_finish:
                conditional_targets.append("FINISH")

            graph_info["edges"].append(
                {
                    "from": "supervisor",
                    "to": "conditional",
                    "type": "conditional",
                    "targets": conditional_targets,
                }
            )

            return graph_info

        except Exception as e:
            self.logger.error(f"Error exporting graph for studio: {str(e)}")
            raise

    def save_graph_visualization(
        self,
        app_id: int,
        parameters: dict | None = None,
        output_path: str | None = None,
    ) -> str:
        """Save graph visualization for debugging and studio integration."""
        try:
            compiled_workflow = self.get_compiled_workflow(app_id, parameters)

            if output_path is None:
                output_path = f"graph_visualization_{app_id}.png"

            try:
                compiled_workflow.get_graph().draw_mermaid_png(output_path=output_path)
                self.logger.info(f"Graph visualization saved to: {output_path}")
                return output_path
            except Exception as viz_error:
                self.logger.warning(f"Could not save graph visualization: {viz_error}")
                return ""

        except Exception as e:
            self.logger.error(f"Error saving graph visualization: {str(e)}")
            raise
