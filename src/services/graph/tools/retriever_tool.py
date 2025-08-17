import asyncio
import os

from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import VectorStoreRetriever

from src.schemas.schema import Model, ModelType
from src.services.document.document_service import DocumentService
from src.services.logger.logger_service import LoggerService

logger = LoggerService().get_logger(__name__)


@tool
def create_vector_retriever_tool(
    retriever: VectorStoreRetriever,
    name: str = "vector_search",
    description: str = "Search for information in the vector database using semantic similarity.",
):
    """Create a retriever tool using LangChain's built-in create_retriever_tool function.

    Args:
        retriever: A VectorStoreRetriever instance
        name: Name of the tool
        description: Description of what the tool does

    Returns:
        A LangChain tool that can be used for vector search
    """
    return create_retriever_tool(
        retriever=retriever,
        name=name,
        description=description,
    )


@tool
async def redis_vector_search_tool(
    question: str, index_name: str = "knowledge_base", search_kwargs: dict = None
) -> list:
    """
    Search in the Redis vector store.
    Parameters:
        question (str): The user question to search for.
    Returns:
        list: A list of documents that match the question.
    """
    logger.debug(
        f"Received question for Redis vector search: {question} with index_name: {index_name} and search_kwargs: {search_kwargs}"
    )

    document_service = DocumentService()

    if search_kwargs is None:
        search_kwargs = {"k": 10, "lambda_mult": 0.5}

    try:

        retriever = await asyncio.to_thread(
            document_service.get_retriever,
            index_name,
            None,
            None,
            search_kwargs,
        )

        results = await retriever.aget_relevant_documents(question)

        if len(results) == 0:
            return ["No relevant documents found."]

        return [doc.page_content for doc in results]

    except Exception as e:
        logger.error(f"Error during Redis vector search: {e}")
        return ["Error occurred during search."]
