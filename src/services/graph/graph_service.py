import asyncio
import importlib
import json
import logging
import os
from functools import partial
from typing import Any, Literal

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
    CheckpointerType,
    GraphConfig,
)
from schemas.topic_validation_schema import TopicValidationRequestSchema
from schemas.personal_data_filter_schema import PersonalDataFilterRequestSchema
from services.chat_history.redis_chat_history import RedisChatHistoryService
from services.data_api.app_settings import AppSettingsService
from services.data_api.chat_history import DataChatHistoryService
from services.validators.topic_validator.topic_validator_service import (
    TopicValidatorService,
)
from services.validators.personal_data.personal_data_filter_service import (
    PersonalDataFilterService,
)
from services.graph.personal_data_filter_checkpointer import PersonalDataFilterCheckpointer
from utils.select_model import get_chat_model


class GraphService:
    """Service for handling multi-agent graph execution with supervisor pattern."""

    def __init__(self, app_settings_service: AppSettingsService):
        self.logger = logging.getLogger(__name__)
        self.app_settings_service = app_settings_service
        self.graph_config: GraphConfig | None = None
        self.workflow = None

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
                f"Loaded graph config with {len(self.graph_config.agents)} agents"
            )

            self.generate_langgraph_studio_config()

        except Exception as ex:
            self.logger.error(f"Error loading graph configuration: {str(ex)}")
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

            self.logger.debug(f"Agent {agent_name} completed processing")
            return state

        except Exception as ex:
            self.logger.error(f"Error in agent {agent_name}: {str(ex)}")

            error_message = AIMessage(
                content=f"Agent {agent_name} encountered an error: {str(ex)}"
            )
            state["messages"] = add_messages(state["messages"], [error_message])
            return state

    def _build_supervisor_prompt(self, available_options: list[str]) -> str:
        """Build the system prompt for the supervisor."""
        agent_descriptions = []
        for name, agent in self.graph_config.agents.items():
            if agent.enabled:
                agent_descriptions.append(f"- {name}: {agent.chain.prompt[:100]}...")

        system_prompt_ending = ""
        if self.graph_config.allow_supervisor_finish:
            system_prompt_ending = "Based on the user input and conversation history, decide which agent should handle this next or if the task is complete (FINISH)."
        else:
            system_prompt_ending = "Based on the user input and conversation history, decide which agent should handle this next. You must select one of the available agents."

        return f"""You are a supervisor managing a team of AI agents. Your job is to decide which agent should handle the user's request{"" if not self.graph_config.allow_supervisor_finish else " or if the task is complete"}.

Available agents and their capabilities:
{chr(10).join(agent_descriptions)}

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
                    raise ValueError(
                        "No enabled agents available and supervisor finish is disabled"
                    )

                available_options = enabled_agents[:]
                if self.graph_config.allow_supervisor_finish:
                    available_options.append("FINISH")

                function_def = self._build_function_definition(available_options)
                system_prompt = self._build_supervisor_prompt(available_options)

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

                self.logger.debug(f"Supervisor decided next action: {state['next']}")
                return state

            except Exception as ex:
                self.logger.error(f"Error in supervisor: {str(ex)}")
                raise

        return supervisor_node

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with supervisor pattern."""
        workflow = StateGraph(AgentState)

        workflow.add_node("topic_validator", self._topic_validator_node)

        workflow.add_node("supervisor", self._create_supervisor_node())

        if self.graph_config.exception_chain:
            workflow.add_node("exception_chain", self._exception_chain_node)

        enabled_agents = {
            name: agent
            for name, agent in self.graph_config.agents.items()
            if agent.enabled 
            and not isinstance(agent, TopicValidationRequestSchema)
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

        def should_continue_from_supervisor(
            state: AgentState,
        ) -> Literal["FINISH"] | str:
            if state["next"] == "FINISH":
                return "FINISH"
            return state["next"]

        def should_continue_from_topic_validator(
            state: AgentState,
        ) -> Literal["supervisor", "exception_chain", "FINISH"]:
            next_step = state.get("next")
            if next_step == "exception_chain":
                return "exception_chain"
            elif next_step == "FINISH":
                return "FINISH"
            else:
                return "supervisor"

        conditional_mapping = {
            agent_name: agent_name for agent_name in enabled_agents.keys()
        }

        if self.graph_config.allow_supervisor_finish:
            conditional_mapping["FINISH"] = END

        workflow.add_conditional_edges(
            "supervisor",
            should_continue_from_supervisor,
            conditional_mapping,
        )

        topic_conditional_mapping = {
            "supervisor": "supervisor",
            "exception_chain": "exception_chain",
            "FINISH": END,
        }

        workflow.add_conditional_edges(
            "topic_validator",
            should_continue_from_topic_validator,
            topic_conditional_mapping,
        )

        workflow.set_entry_point("topic_validator")

        return workflow

    def _create_checkpointer(self):
        """Create checkpointer based on configuration."""
        checkpointer_type = self.graph_config.checkpointer_type
        base_checkpointer = None

        if checkpointer_type == CheckpointerType.MEMORY:
            base_checkpointer = InMemorySaver()
        elif checkpointer_type == CheckpointerType.REDIS:
            try:
                redis_client = redis.Redis(
                    host=os.getenv("REDIS_HOST"),
                    port=int(os.getenv("REDIS_PORT")),
                    db=int(os.getenv("REDIS_DB")),
                    password=os.getenv("REDIS_PASSWORD"),
                    decode_responses=True,
                )
                base_checkpointer = RedisSaver(redis_client)
            except Exception as ex:
                self.logger.warning(
                    f"Failed to create Redis checkpointer: {ex}, falling back to memory"
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
                f"Unknown checkpointer type: {checkpointer_type}, falling back to memory"
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
                logger=self.logger
            )
        
        return base_checkpointer

    async def execute_graph(
        self,
        user_input: str,
        app_id: int,
        user_id: str | None = None,
        context: dict | None = None,
        parameters: dict | None = None,
        save_visualization: bool = False,
    ) -> str:
        """
        Execute the multi-agent graph solution with supervisor pattern.

        Args:
            user_input (str): User input/question
            app_id (str): Application identifier for loading configurations
            user_id (str, optional): User identifier
            context (dict, optional): Additional context
            parameters (dict, optional): Runtime parameters including graph configurations
            save_visualization (bool): Whether to save graph visualization after execution.
                Defaults to False.

        Returns:
            str: Generated response from agents
        """
        self.logger.info(
            f"Graph execution started for app_id: {app_id}, user_id: {user_id}"
        )
        try:
            self.logger.debug(f"Executing graph for user_input: {user_input}")

            user_input = await self._prepare_graph_execution(
                app_id, parameters, user_input
            )

            initial_state = self._create_initial_state(user_input, context, parameters)

            result = await self.workflow.ainvoke(
                initial_state, {"configurable": {"thread_id": user_id}}
            )

            final_response = self._generate_final_response(result)

            if save_visualization:
                try:
                    visualization_path = self.save_graph_visualization(
                        app_id, parameters
                    )
                    if visualization_path:
                        self.logger.info(
                            f"Graph visualization saved during execution: {visualization_path}"
                        )
                except Exception as visualization_error:
                    self.logger.warning(
                        f"Failed to save visualization during execution: {visualization_error}"
                    )

            self.logger.info(
                f"Graph execution completed successfully for app_id: {app_id}"
            )
            return final_response

        except Exception as ex:
            self.logger.error(f"Error executing graph: {str(ex)}")
            return await self._handle_execution_error(user_input, str(ex))

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
                f"User input truncated to {self.graph_config.max_input_length} characters"
            )

        if not self.workflow:
            workflow_builder = self._build_workflow()

            checkpointer = self._create_checkpointer()
            self.workflow = workflow_builder.compile(checkpointer=checkpointer)

        return user_input

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

        self.logger.debug("Graph execution completed successfully")
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

            self.logger.debug("Exception handled by exception chain")
            return response.content

        except Exception as ex:
            self.logger.error(f"Exception chain failed: {str(ex)}")
            raise

    def get_compiled_workflow(self, app_id: int, parameters: dict | None = None):
        """Get the compiled workflow for LangGraph Studio integration."""
        self.logger.info(f"Getting compiled workflow for app_id: {app_id}")
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
                checkpointer = self._create_checkpointer()
                self.workflow = workflow_builder.compile(checkpointer=checkpointer)

            self.logger.info(f"Compiled workflow ready for app_id: {app_id}")
            return self.workflow
        except Exception as ex:
            self.logger.error(f"Error getting compiled workflow: {str(ex)}")
            raise

    def export_graph_for_studio(
        self, app_id: int, parameters: dict | None = None
    ) -> dict:
        """Export graph configuration for LangGraph Studio."""
        self.logger.info(f"Exporting graph for studio for app_id: {app_id}")
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

            self.logger.info(f"Graph export completed for app_id: {app_id}")
            return graph_info

        except Exception as ex:
            self.logger.error(f"Error exporting graph for studio: {str(ex)}")
            raise

    def save_graph_visualization(
        self,
        app_id: int,
        parameters: dict | None = None,
        output_path: str | None = None,
    ) -> str:
        """Save graph visualization for debugging and studio integration.

        Args:
            app_id (int): Application ID for loading graph configuration.
            parameters (dict, optional): Runtime parameters including graph configurations.
            output_path (str, optional): Path where the visualization file will be saved.
                Defaults to "graph_visualization_{app_id}.png".

        Returns:
            str: Path to the saved visualization file, or empty string if failed.
        """
        self.logger.info(f"Saving graph visualization for app_id: {app_id}")
        try:
            compiled_workflow = self.get_compiled_workflow(app_id, parameters)

            if output_path is None:
                output_path = f"graph_visualization_{app_id}.png"

            try:
                compiled_workflow.get_graph().draw_mermaid_png(output_path=output_path)
                self.logger.debug(f"Graph visualization saved to: {output_path}")
                return output_path
            except Exception as visualization_error:
                self.logger.warning(
                    f"Could not save graph visualization: {visualization_error}"
                )
                return ""

        except Exception as ex:
            self.logger.error(f"Error saving graph visualization: {str(ex)}")
            raise

    def generate_langgraph_studio_config(
        self, config_path: str = "langgraph.json"
    ) -> None:
        """Generate or update LangGraph Studio configuration file.

        Creates or updates a LangGraph Studio configuration file that can be used
        for debugging and visualizing the graph workflow. The configuration includes
        the graph path, runtime parameters schema, and default values based on the
        current graph configuration.

        Args:
            config_path (str): Path where the configuration file will be saved.
                Defaults to "langgraph.json" in the project root.

        Raises:
            Exception: If there's an error writing the configuration file or
                if the graph configuration is not properly initialized.
        """
        self.logger.info(f"Generating LangGraph Studio config at: {config_path}")
        try:
            config = {}
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)

            if self.graph_config:
                if "graphs" not in config:
                    config["graphs"] = {}

                if "recruiter_ai_graph" not in config["graphs"]:
                    config["graphs"]["recruiter_ai_graph"] = {
                        "path": "src.services.graph.graph_service:GraphService.get_compiled_workflow"
                    }

                config["graphs"]["recruiter_ai_graph"]["config_schema"] = {
                    "type": "object",
                    "properties": {
                        "app_id": {
                            "type": "integer",
                            "description": "Application ID for loading graph configuration",
                            "default": 0,
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Runtime parameters including graph configurations",
                            "properties": {
                                "graph_config": {
                                    "type": "object",
                                    "description": "Graph configuration object",
                                    "properties": {
                                        "max_input_length": {
                                            "type": "integer",
                                            "description": "Maximum input length allowed",
                                            "default": self.graph_config.max_input_length,
                                        },
                                        "allow_supervisor_finish": {
                                            "type": "boolean",
                                            "description": "Allow supervisor to finish conversation",
                                            "default": self.graph_config.allow_supervisor_finish,
                                        },
                                        "checkpointer_type": {
                                            "type": "string",
                                            "enum": [
                                                "memory",
                                                "redis",
                                                "data",
                                                "custom",
                                            ],
                                            "description": "Type of checkpointer to use",
                                            "default": self.graph_config.checkpointer_type.value,
                                        },
                                        "agents": {
                                            "type": "object",
                                            "description": "Available agents configuration",
                                            "properties": {
                                                agent_name: {
                                                    "type": "object",
                                                    "properties": {
                                                        "enabled": {
                                                            "type": "boolean",
                                                            "description": f"Enable {agent_name} agent",
                                                            "default": agent.enabled,
                                                        }
                                                    },
                                                }
                                                for agent_name, agent in self.graph_config.agents.items()
                                            },
                                        },
                                    },
                                }
                            },
                        },
                    },
                    "required": ["app_id"],
                }

                config["env"] = ".env"

                config["dependencies"] = ["."]

                if "dockerfile_lines" not in config:
                    config["dockerfile_lines"] = [
                        "RUN pip install --no-cache-dir -r requirements.txt"
                    ]

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            self.logger.info(
                f"LangGraph Studio configuration file generated successfully: {config_path}"
            )

        except Exception as ex:
            self.logger.error(f"Error generating LangGraph Studio config: {str(ex)}")
            raise

    def _find_topic_validation_config(self):
        """Find topic validation configuration from agents."""
        for name, agent_config in self.graph_config.agents.items():
            if (
                name == "topic_validator"
                and isinstance(agent_config, TopicValidationRequestSchema)
                and agent_config.enabled
            ):
                return agent_config
        return None

    def _extract_user_input_from_state(self, state: AgentState) -> str:
        """Extract user input from state."""
        user_input = state.get("user_input", "")
        if not user_input and state["messages"]:
            for message in reversed(state["messages"]):
                if hasattr(message, "type") and message.type == "human":
                    user_input = message.content
                    break
        return user_input

    def _handle_invalid_topic(self, state: AgentState, reason: str) -> AgentState:
        """Handle when topic validation fails."""
        if self.graph_config.exception_chain:
            state["next"] = "exception_chain"
            state["error_context"] = f"Topic validation failed: {reason}"
            self.logger.debug(
                f"Question rejected by topic validator, routing to exception chain. Reason: {reason}"
            )
        else:
            error_message = AIMessage(
                content="I can only help with work-related topics. Please ask questions about careers, job search, professional development, or workplace matters."
            )
            state["messages"] = add_messages(state["messages"], [error_message])
            state["next"] = "FINISH"
            self.logger.debug(
                f"Question rejected by topic validator, no exception chain available. Reason: {reason}"
            )
        return state

    def _find_personal_data_filter_config(self):
        """Find personal data filter configuration from agents."""
        for name, agent_config in self.graph_config.agents.items():
            if (
                name == "personal_data_filter"
                and isinstance(agent_config, PersonalDataFilterRequestSchema)
                and agent_config.enabled
            ):
                return agent_config
        return None

    async def _topic_validator_node(self, state: AgentState) -> AgentState:
        """Topic validator node logic."""
        try:
            topic_validation_config = self._find_topic_validation_config()

            if not topic_validation_config:
                self.logger.debug("Topic validation is disabled, skipping validation")
                return state

            user_input = self._extract_user_input_from_state(state)

            if not user_input:
                self.logger.warning("No user input found for topic validation")
                return state

            topic_validator_service = TopicValidatorService()
            is_valid, topic, reason = await topic_validator_service.validate_topic(
                question=user_input
            )

            self.logger.debug(
                f"Topic validation result: valid={is_valid}, topic={topic}, reason={reason}"
            )

            if not is_valid:
                return self._handle_invalid_topic(state, reason)

            self.logger.debug("Topic validation passed, proceeding to supervisor")
            return state

        except Exception as ex:
            self.logger.error(f"Error in topic validator: {str(ex)}")
            return state

    async def _exception_chain_node(self, state: AgentState) -> AgentState:
        """Exception chain node for handling errors and invalid inputs."""
        try:
            user_input = self._extract_user_input_from_state(state)
            error_context = state.get("error_context", "General error occurred")
            
            llm = await get_chat_model(
                provider=self.graph_config.exception_chain.chain.model.provider.value,
                deployment=self.graph_config.exception_chain.chain.model.deployment,
                model=self.graph_config.exception_chain.chain.model.name,
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=f"{self.graph_config.exception_chain.chain.prompt}\n\nUser input: {user_input}\nContext: {error_context}"
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )

            chain = prompt | llm
            response = await chain.ainvoke({"messages": state["messages"]})
            
            state["messages"] = add_messages(state["messages"], [response])
            state["next"] = "FINISH"
            
            self.logger.debug("Exception chain handled the error")
            return state

        except Exception as ex:
            self.logger.error(f"Error in exception chain: {str(ex)}")
            error_message = AIMessage(
                content="I apologize, but I'm unable to process your request at this time. Please try again later."
            )
            state["messages"] = add_messages(state["messages"], [error_message])
            state["next"] = "FINISH"
            return state
