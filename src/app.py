from fastapi import FastAPI
from dotenv.main import load_dotenv, find_dotenv
import uvicorn
import os

from services.logger.logger_service import LoggerService
from config import Config

from routes.agent_routes import router as question_answer_router
from routes.vector_db_routes import router as vector_db_router


def create_app():
    os.environ.clear()
    load_dotenv(find_dotenv())
    
    app = FastAPI()

    app.include_router(question_answer_router)
    app.include_router(vector_db_router)

    LoggerService().setup_logger(Config.LOG_LEVEL)

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("app:app", reload=False, port=int(Config.PORT), host="0.0.0.0")
