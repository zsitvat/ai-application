import os

import numpy as np
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import VectorStoreRetriever
from redis import Redis

from src.services.logger.logger_service import LoggerService
from src.utils.select_model import get_embedding_model

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
def redis_vector_search_tool(
    question: str,
    index_name: str = "knowledge_base",
    k: int = 10,
    embedding_field: str = "embedding",
) -> list:
    """
    Vektor alapú keresés a Redis FT indexben.
    Parameters:
        question (str): A keresett kérdés.
        index_name (str): Az index neve.
        k (int): Találatok száma.
        embedding_field (str): Az embedding mező neve az indexben.
    Returns:
        list: A releváns dokumentumok szövege.
    """
    try:

        embedding_model = get_embedding_model(
            provider=os.getenv("EMBEDDING_PROVIDER", ""),
            deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME", ""),
            model=os.getenv("EMBEDDING_MODEL", ""),
        )
        logger.info(f"[REDIS VECTOR] Embedding model: {embedding_model}")

        embedding = embedding_model.embed_query(question)
        if embedding is None:
            logger.error("Nem sikerült embeddinget generálni a kérdéshez.")
            return ["Embedding error: Nem sikerült embeddinget generálni a kérdéshez."]

        redis = Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            username=os.getenv("REDIS_USER"),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=False,
        )

        base_query = "*"
        query_params = {"vec": np.array(embedding, dtype=np.float32).tobytes()}
        query_str = f"{base_query}=>[KNN {k} @{embedding_field} $vec AS {embedding_field}_score]"
        logger.info(f"[REDIS VECTOR] Query: {query_str}")

        results = redis.ft(index_name).search(query_str, query_params)

        logger.info(f"[REDIS VECTOR] Találatok száma: {results.total}")
        if results.total == 0:
            logger.warning(f"Nincs releváns dokumentum a kérdésre: {question}")
            return ["No relevant documents found."]
        output = []
        for doc in results.docs:
            text = getattr(doc, "text", b"")
            if isinstance(text, bytes):
                output.append(text.decode("utf-8", errors="ignore"))
            else:
                output.append(str(text))
        return output
    except Exception as e:
        logger.error(f"[RetrieverTool] Error during Redis vector search: {e}")
        logger.error(f"[RetrieverTool] Error type: {type(e).__name__}")
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
