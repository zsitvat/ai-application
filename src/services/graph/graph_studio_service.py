import json
import logging
import os
from typing import Any

from .graph_service import GraphService


class GraphStudioService:
    """Service for handling LangGraph Studio integration and visualization."""

    def __init__(self, graph_service: GraphService = None):
        self.logger = logging.getLogger(__name__)
        self.graph_service = graph_service

    def export_graph_for_studio(
        self, app_id: int, parameters: dict | None = None
    ) -> dict:
        """Export graph configuration for LangGraph Studio.

        Args:
            app_id (int): Application ID
            parameters (dict, optional): Runtime parameters

        Returns:
            dict: Graph information for studio
        """
        self.logger.info(f"Exporting graph for studio for app_id: {app_id}")
        try:
            if not self.graph_service:
                raise ValueError("GraphService not available")

            self.graph_service.get_compiled_workflow(app_id, parameters)
            graph_config = self.graph_service.graph_config

            if not graph_config:
                raise ValueError("Graph configuration not loaded")

            graph_info = {
                "nodes": [],
                "edges": [],
                "config": {
                    "checkpointer_enabled": graph_config.enable_checkpointer,
                    "checkpointer_type": (
                        graph_config.checkpointer_type.value
                        if graph_config.checkpointer_type
                        else None
                    ),
                    "max_input_length": graph_config.max_input_length,
                    "agents_count": len(graph_config.agents),
                    "allow_supervisor_finish": graph_config.allow_supervisor_finish,
                },
            }

            enabled_agents = {
                name: agent
                for name, agent in graph_config.agents.items()
                if agent.enabled
            }

            graph_info["nodes"].append(
                {
                    "id": "supervisor",
                    "type": "supervisor",
                    "config": {
                        "model": {
                            "provider": graph_config.supervisor.chain.model.provider.value,
                            "name": graph_config.supervisor.chain.model.name,
                            "deployment": graph_config.supervisor.chain.model.deployment,
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
            if graph_config.allow_supervisor_finish:
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
            if not self.graph_service:
                raise ValueError("GraphService not available")

            compiled_workflow = self.graph_service.get_compiled_workflow(
                app_id, parameters
            )

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
            if not self.graph_service:
                raise ValueError("GraphService not available")

            graph_config = self.graph_service.graph_config

            config = {}
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)

            if graph_config:
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
                                            "default": graph_config.max_input_length,
                                        },
                                        "allow_supervisor_finish": {
                                            "type": "boolean",
                                            "description": "Allow supervisor to finish conversation",
                                            "default": graph_config.allow_supervisor_finish,
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
                                            "default": graph_config.checkpointer_type.value,
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
                                                for agent_name, agent in graph_config.agents.items()
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
