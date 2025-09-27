import asyncio

from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import VectorStoreRetriever

from src.services.document.document_service import DocumentService
from src.services.logger.logger_service import LoggerService

logger = LoggerService().setup_logger()


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
    question: str, index_name: str = "knowledge_base", search_kwargs: dict | None = None
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
        logger.debug(f"Attempting to get retriever for index: {index_name}")

        retriever = await asyncio.to_thread(
            document_service.get_retriever,
            index_name,
            None,
            None,
            search_kwargs,
        )

        logger.debug(f"Successfully got retriever, now searching for: {question}")
        results = await retriever.aget_relevant_documents(question)

        if len(results) == 0:
            logger.warning(f"No relevant documents found for question: {question}")
            return ["No relevant documents found."]

        logger.debug(f"Found {len(results)} relevant documents")
        return [doc.page_content for doc in results]

    except Exception as e:
        logger.error(f"[RetrieverTool] Error during Redis vector search: {e}")
        logger.error(f"[RetrieverTool] Error type: {type(e).__name__}")
        logger.error(
            f"[RetrieverTool] Search parameters: index_name={index_name}, search_kwargs={search_kwargs}"
        )

        if "Connection" in str(e) or "redis" in str(e).lower():
            return [
                f"Redis connection error: {str(e)}. Please check if Redis is running and accessible."
            ]
        elif "index" in str(e).lower() or "unknown index" in str(e).lower():
            return [f"Index '{index_name}' not found or not accessible: {str(e)}"]
        elif "embedding" in str(e).lower():
            return [
                f"Embedding model error: {str(e)}. Please check embedding provider configuration."
            ]
        else:
            return [f"Search error: {str(e)}"]
