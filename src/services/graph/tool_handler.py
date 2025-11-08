import asyncio
import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from src.schemas.graph_schema import Agent, AgentState
from src.services.graph.tools.tools_config import AVAILABLE_TOOLS


class ToolHandler:
    """Handles all tool-related operations for the Graph service."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_allowed_tool_names(self, agent_config: Agent) -> list[str]:
        """Get list of allowed tool names for an agent, excluding required tools."""
        allowed_tool_names = []
        if hasattr(agent_config, "tools") and isinstance(agent_config.tools, dict):
            for tool_name, tool_config in agent_config.tools.items():
                # Skip required tools, because they are executed automatically
                if tool_config.get("required", False):
                    continue
                if tool_name in AVAILABLE_TOOLS:
                    allowed_tool_names.append(tool_name)
        return allowed_tool_names

    def bind_tools_to_chain(self, agent_config: Agent) -> list[Any]:
        """Bind tools to the agent chain based on the agent configuration."""
        allowed_tool_names = self.get_allowed_tool_names(agent_config)
        tools_to_bind = []

        for name in allowed_tool_names:
            tool_func = AVAILABLE_TOOLS[name]
            if name == "get_position_tool" and name in agent_config.tools:
                input_fields = agent_config.tools[name].get("input_fields", [])
                tool_func = tool_func(input_fields if input_fields else [])
                self.logger.debug(
                    f"[ToolHandler] Created position tool for binding with input_fields: {input_fields if input_fields else '[]'}"
                )
            tools_to_bind.append(tool_func)
        return tools_to_bind

    def extract_tool_call(
        self, response: Any
    ) -> tuple[str | None, dict | None, str | None]:
        """Extract tool call information from LLM response."""
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
                self.logger.warning(f"[ToolHandler] Error parsing tool_calls: {e}")
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
                self.logger.warning(f"[ToolHandler] Error parsing function_call: {e}")
                tool_name = None
                tool_args = None
                tool_call_id = None

        return tool_name, tool_args, tool_call_id

    def find_tool_func(self, tool_name: str | None, tools_to_bind: list[Any]) -> Any:
        """Find a tool function by name in the given list of tools."""
        if tool_name:
            for t in tools_to_bind:
                if hasattr(t, "name") and t.name == tool_name:
                    return t
        return AVAILABLE_TOOLS.get(tool_name)

    def merge_tool_args_with_config(
        self, tool_args: dict, config_defaults: dict
    ) -> dict:
        """Merge tool arguments with configuration defaults."""
        merged_args = dict(tool_args)
        for k, v in config_defaults.items():
            merged_args[k] = v
        return merged_args

    def prepare_tool_args(
        self, tool_name: str, tool_args: dict, agent_config: Agent
    ) -> dict:
        """Prepare tool arguments for the given tool."""
        if (
            isinstance(tool_args, dict)
            and "input_fields" in tool_args
            and isinstance(tool_args["input_fields"], dict)
        ):
            search_values = tool_args["input_fields"]
            tool_args = search_values.copy()

        if tool_name in agent_config.tools and isinstance(tool_args, dict):
            tool_args = self.merge_tool_args_with_config(
                tool_args, agent_config.tools[tool_name]
            )
            self.logger.debug(
                f"[ToolHandler] Tool args after merge for '{tool_name}': {tool_args}"
            )

        return tool_args

    async def execute_tool_function(
        self,
        tool_func: Any,
        tool_name: str,
        tool_args: dict,
        agent_config: Agent,
        app_id: int | None = None,
    ) -> ToolMessage:
        """Execute a tool function with the given arguments."""
        if not (tool_func and callable(tool_func)):
            self.logger.error(
                f"[ToolHandler] tool_func for '{tool_name}' is not callable: {tool_func}"
            )
            return ToolMessage(
                content=f"Error: tool '{tool_name}' is not callable.",
                tool_call_id=tool_name or None,
            )

        try:

            tool_func = self._handle_position_tool_special_case(
                tool_func, tool_name, agent_config
            )

            filtered_args = self._prepare_filtered_args(tool_args, app_id)

            tool_result = await self._execute_tool_by_type(
                tool_func, filtered_args, tool_name
            )

            return self._wrap_tool_result(tool_result, tool_name)

        except Exception as ex:
            self.logger.error(
                f"[ToolHandler] Error executing tool '{tool_name}': {str(ex)}"
            )
            return ToolMessage(
                content=f"Error executing tool '{tool_name}': {str(ex)}",
                tool_call_id=tool_name or None,
            )

    def _handle_position_tool_special_case(
        self, tool_func: Any, tool_name: str, agent_config: Agent
    ) -> Any:
        """Handle special case for position tool with input_fields."""
        if tool_name == "get_position_tool" and tool_name in agent_config.tools:
            input_fields = agent_config.tools[tool_name].get("input_fields", [])
            if input_fields:
                tool_func = tool_func(input_fields)
                self.logger.debug(
                    f"[ToolHandler] Created position tool with input_fields: {input_fields}"
                )
        return tool_func

    def _prepare_filtered_args(self, tool_args: dict, app_id: int | None) -> dict:
        """Prepare filtered arguments for tool execution."""
        config_keys = {
            "type",
            "job_type",
            "input_fields",
            "description",
            "required",
            "variable_name",
        }
        filtered_args = {k: v for k, v in tool_args.items() if k not in config_keys}
        filtered_args["app_id"] = app_id
        return filtered_args

    async def _execute_tool_by_type(
        self, tool_func: Any, filtered_args: dict, tool_name: str
    ) -> Any:
        """Execute tool based on its type (async, sync, etc.)."""
        self.logger.debug(
            f"[ToolHandler] Executing tool '{tool_name}' with filtered_args: {filtered_args}"
        )

        if hasattr(tool_func, "ainvoke"):
            return await tool_func.ainvoke(filtered_args)
        elif asyncio.iscoroutinefunction(tool_func):
            return await tool_func(filtered_args)
        else:
            return tool_func(filtered_args)

    def _wrap_tool_result(self, tool_result: Any, tool_name: str) -> ToolMessage:
        """Wrap tool result in appropriate message format."""
        if isinstance(tool_result, ToolMessage):
            return tool_result
        elif isinstance(tool_result, AIMessage):
            return ToolMessage(
                content=tool_result.content, tool_call_id=tool_name or None
            )
        else:
            return ToolMessage(content=str(tool_result), tool_call_id=tool_name or None)

    async def execute_required_tools(self, agent_config: Agent) -> list[dict[str, Any]]:
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
                f"[ToolHandler] Processing required tool '{tool_name}' for agent"
            )

            tool_func = AVAILABLE_TOOLS.get(tool_name)
            if not tool_func:
                self.logger.warning(
                    f"[ToolHandler] Required tool '{tool_name}' not found in AVAILABLE_TOOLS."
                )
                continue

            try:
                tool_args = dict(tool_config)
                tool_result = await self.execute_tool_function(
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

            except Exception as ex:
                self.logger.error(
                    f"[ToolHandler] Failed to execute required tool '{tool_name}': {str(ex)}"
                )

        return required_tool_results

    async def run_tool_and_update_state(
        self,
        state: AgentState,
        tool_func: Any,
        tool_name: str,
        tool_args: dict,
        agent_config: Agent,
        tool_call_id: str | None,
        prompt_context: dict,
        update_context_callback: callable,
        filter_messages_callback: callable,
    ) -> bool:
        """Run the tool function and update the state with the result."""
        tool_result_message = await self.execute_tool_function(
            tool_func,
            tool_name,
            tool_args,
            agent_config,
            app_id=state.parameters.get("app_id"),
        )

        if tool_result_message:
            if isinstance(tool_result_message, ToolMessage):
                tool_result_message.tool_call_id = (
                    tool_call_id or tool_name or "unknown_tool"
                )

            state["messages"] = state["messages"] + [tool_result_message]

            var_name = None
            if hasattr(agent_config, "tools") and tool_name in agent_config.tools:
                var_name = agent_config.tools[tool_name].get("variable_name")

            update_context_callback(
                prompt_context,
                [
                    {
                        "tool_name": tool_name,
                        "message": tool_result_message,
                        "variable_name": var_name,
                    }
                ],
                state.application_attributes,
                state=state,
            )

            prompt_context["messages"] = filter_messages_callback(state.messages)
            return True
        return False

    def inject_tool_info_into_prompt(self, prompt: Any, agent_config: Agent) -> Any:
        """Inject tool information into the prompt."""
        if hasattr(agent_config, "tools") and isinstance(agent_config.tools, dict):
            all_tool_info = []
            for tool_name, tool_conf in agent_config.tools.items():
                tool_info_lines = [f"Tool: {tool_name}"]
                for k, v in tool_conf.items():
                    if isinstance(v, (list, tuple)):
                        tool_info_lines.append(f"  {k}: {', '.join(map(str, v))}")
                    else:
                        tool_info_lines.append(f"  {k}: {v}")
                all_tool_info.append("\n".join(tool_info_lines))

            if all_tool_info:
                prompt = prompt.partial(agent_tool_info="\n\n".join(all_tool_info))
        return prompt

    def update_context_with_tool_results(
        self,
        prompt_context: dict,
        required_tools_executed: list[dict],
        application_attributes: dict | None = None,
        state: AgentState | None = None,
    ) -> None:
        """Update prompt context and state.context with data from required tools results."""
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

            if state is not None and tool_name:
                state.context[tool_name] = content

            if application_attributes is not None and content:
                self._update_application_attributes_from_tool(
                    application_attributes, tool_name, content
                )

    def _update_prompt_context(
        self,
        prompt_context: dict,
        tool_name: str,
        var_name: str | None,
        content: str,
        tool_message: Any,
    ) -> None:
        """Update prompt context with tool result."""
        if var_name:
            prompt_context[var_name] = content or str(tool_message)
        elif tool_name == "get_labels_tool":
            prompt_context["labels"] = content or str(tool_message)

    def _update_application_attributes_from_tool(
        self, application_attributes: dict, tool_name: str, content: str
    ) -> None:
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
                f"[ToolHandler] Updated application_attributes from required tool '{tool_name}'"
            )

        except (json.JSONDecodeError, TypeError) as ex:
            self.logger.debug(
                f"[ToolHandler] Could not parse tool result as JSON for '{tool_name}': {ex}"
            )
            self._process_text_tool_data(application_attributes, tool_name, content)

    def _process_structured_tool_data(
        self, application_attributes: dict, tool_name: str, parsed_content: Any
    ) -> None:
        """Process structured JSON data from tools."""
        if tool_name == "get_positions_tool" and isinstance(parsed_content, list):
            self._extract_position_info_from_list(
                application_attributes, parsed_content
            )
        elif tool_name == "get_labels_tool" and isinstance(parsed_content, dict):
            self._extract_location_info_from_labels(
                application_attributes, parsed_content
            )

    def _extract_position_info_from_list(
        self, application_attributes: dict, positions: list
    ) -> None:
        """Extract position information from positions data."""
        for position in positions:
            if isinstance(position, dict):
                if "id" in position and not application_attributes.get("position_id"):
                    application_attributes["position_id"] = str(position["id"])
                if "title" in position and not application_attributes.get(
                    "position_name"
                ):
                    application_attributes["position_name"] = position["title"]
                break

    def _extract_location_info_from_labels(
        self, application_attributes: dict, labels_data: dict
    ) -> None:
        """Extract location information from labels."""
        if "counties" in labels_data:
            counties = labels_data["counties"]
            if counties and not application_attributes.get("other_information"):
                application_attributes["other_information"] = (
                    f"Available locations: {', '.join(counties[:5])}"
                )

    def _process_text_tool_data(
        self, application_attributes: dict, tool_name: str, content: str
    ) -> None:
        """Process text content from tools."""
        if tool_name == "get_positions_tool" and "position" in content.lower():
            self._extract_position_info_from_text(application_attributes, content)

    def _extract_position_info_from_text(
        self, application_attributes: dict, content: str
    ) -> None:
        """Extract position info from text content."""
        lines = content.split("\n")
        for line in lines[:3]:
            if "id:" in line.lower() and not application_attributes.get("position_id"):
                try:
                    application_attributes["position_id"] = line.split(":", 1)[
                        1
                    ].strip()
                except (IndexError, AttributeError):
                    pass
            elif "title:" in line.lower() and not application_attributes.get(
                "position_name"
            ):
                try:
                    application_attributes["position_name"] = line.split(":", 1)[
                        1
                    ].strip()
                except (IndexError, AttributeError):
                    pass
