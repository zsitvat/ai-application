import os

import uvicorn
import logging
from dotenv.main import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langfuse import Langfuse

from src.routes.dataset_routes import router as dataset_router
from src.routes.document_routes import router as document_router
from src.routes.graph_config_loader_routes import router as graph_config_loader_router
from src.routes.graph_routes import router as graph_router
from src.routes.personal_data_filter_routes import router as personal_data_filter_router
from src.routes.system_routes import router as system_router
from src.routes.topic_validation_routes import router as topic_validation_router
from src.routes.web_scraping_routes import router as web_scraping_router
from src.services.logger.logger_service import LoggerService
from src.services.rate_limit.semaphore import SemaphoreMiddleware
from src.services.logger.logs_json_formatter import JSONFormatter

if os.getenv("TRACER_TYPE") == "langfuse":
    langfuse = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        project_id=os.getenv("LANGFUSE_PROJECT"),
        host=os.getenv("LANGFUSE_ENDPOINT"),
    )


def create_app():
    os.environ.clear()
    load_dotenv(find_dotenv())

    app = FastAPI(
        title="Recruiter AI API",
        description="Multi-agent AI system app with document processing and web scraping capabilities",
    )

    app.include_router(graph_router)
    app.include_router(graph_config_loader_router)
    app.include_router(personal_data_filter_router)
    app.include_router(topic_validation_router)
    app.include_router(web_scraping_router)
    app.include_router(document_router)
    app.include_router(dataset_router)
    app.include_router(system_router)

    @app.get("/")
    async def redirect_to_docs():
        return RedirectResponse(url="/docs")

    LoggerService().setup_logger(os.getenv("LOG_LEVEL", "INFO"))

    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            handler.setFormatter(JSONFormatter())

    default_limit = int(os.getenv("RATELIMIT", "20"))
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

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "app:app", reload=False, port=int(os.getenv("PORT", 5000)), host="0.0.0.0"
    )
