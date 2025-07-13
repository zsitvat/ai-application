import json
import logging
from pathlib import Path
from typing import Union

import aiofiles
import aiohttp
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, HttpUrl

from src.schemas.graph_schema import RestOperationPostSchema
from src.services.data_api.app_settings import AppSettingsService
from src.services.graph.graph_service import GraphService

router = APIRouter(tags=["Graph Config Loader"])


class GraphConfigLoaderSchema(BaseModel):
    config_source: Union[str, HttpUrl]
    uuid: str
    applicationIdentifier: dict
    platform: str = "webchat"
    user_input: str
    context: dict = {}
    user_id: str | None = None


def get_graph_service():
    app_settings_service = AppSettingsService()
    return GraphService(app_settings_service)


async def load_config_from_file(file_path: str) -> dict:
    """Load configuration from local file."""
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        if path.suffix.lower() != ".json":
            raise ValueError("Configuration file must be a JSON file")

        async with aiofiles.open(path, "r", encoding="utf-8") as file:
            content = await file.read()
            return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    except Exception as e:
        raise ValueError(f"Error reading configuration file: {e}")


async def load_config_from_url(url: str) -> dict:
    """Load configuration from web URL."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise ValueError(
                        f"Failed to fetch configuration from URL. Status: {response.status}"
                    )

                content_type = response.headers.get("content-type", "")
                if "application/json" not in content_type:
                    raise ValueError("URL must return JSON content")

                content = await response.text()
                return json.loads(content)
    except aiohttp.ClientError as e:
        raise ValueError(f"Network error fetching configuration: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from URL: {e}")
    except Exception as e:
        raise ValueError(f"Error fetching configuration from URL: {e}")


@router.post("/api/graph/execute-with-config")
async def execute_graph_with_config(
    request: GraphConfigLoaderSchema,
    graph_service: GraphService = Depends(get_graph_service),
):
    """
    Execute graph with configuration loaded from file path or web URL.

    Loads a graph configuration from either local file path or web URL,
    then executes the graph with the loaded configuration.
    """
    logger = logging.getLogger(__name__)

    try:
        config_source = str(request.config_source)
        logger.info(f"Loading graph configuration from: {config_source}")

        if config_source.startswith(("http://", "https://")):
            graph_config = await load_config_from_url(config_source)
            logger.debug(f"Loaded configuration from URL: {config_source}")
        else:
            graph_config = await load_config_from_file(config_source)
            logger.debug(f"Loaded configuration from file: {config_source}")

        if not isinstance(graph_config, dict):
            raise ValueError("Configuration must be a JSON object")

        if "agents" not in graph_config:
            raise ValueError("Configuration must contain 'agents' section")

        if "supervisor" not in graph_config:
            raise ValueError("Configuration must contain 'supervisor' section")

        parameters = {"graph_config": graph_config}

        response = await graph_service.execute_graph(
            user_input=request.user_input,
            app_id=0,
            user_id=request.user_id,
            context=request.context,
            parameters=parameters,
        )

        logger.info(
            f"Graph execution completed successfully with config from: {config_source}"
        )

        return {
            "response": response,
            "config_source": config_source,
            "status": "success",
        }

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in graph execution with config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/api/graph/validate-config")
async def validate_graph_config(config_source: str):
    """
    Validate a graph configuration from file path or URL without executing it.

    Returns validation result and basic configuration info including
    agent count, enabled agents, and configuration structure.
    """
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Validating graph configuration from: {config_source}")

        if config_source.startswith(("http://", "https://")):
            graph_config = await load_config_from_url(config_source)
        else:
            graph_config = await load_config_from_file(config_source)

        if not isinstance(graph_config, dict):
            raise ValueError("Configuration must be a JSON object")

        required_fields = ["agents", "supervisor"]
        missing_fields = [
            field for field in required_fields if field not in graph_config
        ]

        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        agents = graph_config.get("agents", {})
        enabled_agents = [
            name for name, config in agents.items() if config.get("enabled", True)
        ]

        config_info = {
            "total_agents": len(agents),
            "enabled_agents": len(enabled_agents),
            "agent_names": list(agents.keys()),
            "enabled_agent_names": enabled_agents,
            "has_supervisor": "supervisor" in graph_config,
            "has_exception_chain": "exception_chain" in graph_config,
            "checkpointer_type": graph_config.get("checkpointer_type", "memory"),
            "max_input_length": graph_config.get("max_input_length", -1),
        }

        logger.info(f"Configuration validation successful for: {config_source}")

        return {
            "valid": True,
            "config_source": config_source,
            "config_info": config_info,
            "status": "valid",
        }

    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        return {
            "valid": False,
            "config_source": config_source,
            "error": str(e),
            "status": "invalid",
        }
