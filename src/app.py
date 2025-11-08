import logging
from typing import Optional

import uvicorn
from dotenv.main import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from langfuse import Langfuse

from src.config.app_config import config
from src.routes.dataset_routes import router as dataset_router
from src.routes.document_routes import router as document_router
from src.routes.file_upload_routes import router as file_upload_router
from src.routes.graph_config_loader_routes import router as graph_config_loader_router
from src.routes.graph_routes import router as graph_router
from src.routes.personal_data_filter_routes import router as personal_data_filter_router
from src.routes.system_routes import router as system_router
from src.routes.topic_validation_routes import router as topic_validation_router
from src.routes.web_scraping_routes import router as web_scraping_router
from src.services.logger.logger_service import LoggerService
from src.services.logger.logs_json_formatter import JSONFormatter
from src.services.rate_limit.semaphore import SemaphoreMiddleware


def _initialize_langfuse() -> Optional[Langfuse]:
    """Initialize Langfuse if TRACER_TYPE is set to langfuse."""
    if config.tracing.tracer_type == "langfuse":
        return Langfuse(
            secret_key=config.tracing.langfuse_secret_key,
            public_key=config.tracing.langfuse_public_key,
            project_id=config.tracing.langfuse_project,
            host=config.tracing.langfuse_endpoint,
        )
    return None


def _setup_logging() -> None:
    """Setup application logging with JSON formatter."""
    LoggerService().setup_logger(config.logging.level)

    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            handler.setFormatter(JSONFormatter())


def _setup_middleware(app: FastAPI) -> None:
    """Setup middleware for the FastAPI application."""

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:2024",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:2024",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    default_limit = config.rate_limit
    paths_to_limit = [
        "/api/graph",
        "/api/graph/stream",
        "/api/health-check",
    ]
    app.add_middleware(
        SemaphoreMiddleware,
        semaphore_limit=default_limit,
        paths_to_limit=paths_to_limit,
    )


langfuse = _initialize_langfuse()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    load_dotenv(find_dotenv())

    app = FastAPI(
        title="Recruiter AI API",
        description="Multi-agent AI system app with document processing and web scraping capabilities",
    )

    routers = [
        graph_router,
        graph_config_loader_router,
        personal_data_filter_router,
        topic_validation_router,
        web_scraping_router,
        document_router,
        dataset_router,
        system_router,
        file_upload_router,
    ]

    for router in routers:
        app.include_router(router)

    @app.get("/")
    async def redirect_to_docs():
        """Redirect root path to API documentation."""
        return RedirectResponse(url="/docs")

    _setup_logging()
    _setup_middleware(app)

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        reload=False,
        port=config.port,
        host="0.0.0.0",
    )
