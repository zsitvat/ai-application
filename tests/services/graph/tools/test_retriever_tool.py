import pytest
from unittest.mock import MagicMock
from src.services.graph.tools import retriever_tool


def test_create_vector_retriever_tool():
    tool = retriever_tool.create_vector_retriever_tool
    assert hasattr(tool, "name")
    assert hasattr(tool, "description")


@pytest.mark.asyncio
async def test_redis_vector_search_tool_success():
    class DummyRetriever:
        async def aget_relevant_documents(self, question):
            return [MagicMock(page_content="doc1"), MagicMock(page_content="doc2")]

    class DummyDocumentService:
        def get_retriever(self, *args, **kwargs):
            return DummyRetriever()

    retriever_tool.DocumentService = DummyDocumentService
    tool = retriever_tool.redis_vector_search_tool
    result = await tool.ainvoke({"question": "test question"})
    assert result == ["doc1", "doc2"]


@pytest.mark.asyncio
async def test_redis_vector_search_tool_error():
    class DummyDocumentService:
        def get_retriever(self, *args, **kwargs):
            raise Exception("fail")

    retriever_tool.DocumentService = DummyDocumentService
    tool = retriever_tool.redis_vector_search_tool
    result = await tool.ainvoke({"question": "test question"})
    assert result == ["Error occurred during search."]
