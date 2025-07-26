from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import VectorStoreRetriever

import os
import asyncio
from src.services.document.document_service import DocumentService
from schemas.model_schema import Model
from langchain_core.tools import tool


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
def redis_vector_search_tool(index_name: str, query: str, k: int = 5) -> list:
    """Keresés a Redis vector store-ban. Paraméterek: index_name (str), query (str), k (int, top-k találat)."""
    document_service = DocumentService()
    model = Model(
        model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        model_type=os.getenv("EMBEDDING_MODEL_TYPE", "embedding"),
    )
    retriever = asyncio.run(
        document_service.get_retriever(
            index_name=index_name,
            model=model,
            index_schema=None,
            search_kwargs={"k": k},
        )
    )
    results = asyncio.run(retriever.aget_relevant_documents(query))
    return [doc.page_content for doc in results]
