import asyncio
import os

from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import VectorStoreRetriever

from schemas.schema import Model
from services.document.document_service import DocumentService


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
    index_name: str, query: str, search_kwargs: dict = None
) -> list:
    """
    Keresés a Redis vector store-ban. Paraméterek: index_name (str), query (str), search_kwargs (dict, pl. k, lambda_mult, stb.).
    """
    document_service = DocumentService()
    model = Model(
        name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        type=os.getenv("EMBEDDING_MODEL_TYPE", "embedding"),
        deployment=os.getenv(
            "EMBEDDING_DEPLOYMENT",
            "chatboss_sweden-central_embedding_text-embedding-3-large_1",
        ),
        provider=os.getenv("EMBEDDING_PROVIDER", "azure"),
    )

    if search_kwargs is None:
        search_kwargs = {"k": 5, "lambda_mult": 0.5}

    retriever = asyncio.run(
        document_service.get_retriever(
            index_name=index_name,
            model=model,
            index_schema=None,
            search_kwargs=search_kwargs,
        )
    )

    results = asyncio.run(retriever.aget_relevant_documents(query))
    return [doc.page_content for doc in results]
