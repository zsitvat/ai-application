from fastapi import FastAPI
from dotenv.main import load_dotenv, find_dotenv
import uvicorn
import os

from services.logger.logger_service import LoggerService
from langfuse import Langfuse

from routes.graph_routes import router as graph_router
from routes.personal_data_filter_routes import router as personal_data_filter_router
from routes.topic_validation_routes import router as topic_validation_router
from routes.web_scraping_routes import router as web_scraping_router
from routes.document_routes import router as document_router
from routes.dataset_routes import router as dataset_router
from routes.system_routes import router as system_router
from services.rate_limit.semaphore import SemaphoreMiddleware


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
    app.include_router(personal_data_filter_router)
    app.include_router(topic_validation_router)
    app.include_router(web_scraping_router)
    app.include_router(document_router)
    app.include_router(dataset_router)
    app.include_router(system_router)

    LoggerService().setup_logger(os.getenv("LOG_LEVEL", "INFO"))

    default_limit = int(os.getenv("RATELIMIT", "5"))
    paths_to_limit = [
        "/api/graph",
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
