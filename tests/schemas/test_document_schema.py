from src.schemas.document_schema import (
    DocumentDeleteResponseSchema,
    DocumentIngestRequestSchema,
    DocumentIngestResponseSchema,
    PositionsIngestRequestSchema,
)


def test_document_ingest_request_schema():
    """Test DocumentIngestRequestSchema instantiation and field values."""
    obj = DocumentIngestRequestSchema(
        vector_db_index="index1",
        files=["file1.txt", "file2.txt"],
        chunk_size=8000,
        chunk_overlap=300,
        json_data=[{"content": "test"}],
        embedding_model=None,
        index_schema=None,
    )
    assert obj.vector_db_index == "index1"
    assert isinstance(obj.files, list)
    assert obj.chunk_size == 8000
    assert obj.chunk_overlap == 300
    assert isinstance(obj.json_data, list)
    assert obj.embedding_model is None
    assert obj.index_schema is None


def test_document_ingest_response_schema():
    """Test DocumentIngestResponseSchema instantiation and field values."""
    obj = DocumentIngestResponseSchema(
        success=True,
        processed_files=["file1.txt"],
        failed_files=["file2.txt"],
        message="Done",
    )
    assert obj.success is True
    assert obj.processed_files == ["file1.txt"]
    assert obj.failed_files == ["file2.txt"]
    assert obj.message == "Done"


def test_document_delete_response_schema():
    """Test DocumentDeleteResponseSchema instantiation and field values."""
    obj = DocumentDeleteResponseSchema(success=True, message="Deleted", deleted_count=2)
    assert obj.success is True
    assert obj.message == "Deleted"
    assert obj.deleted_count == 2


def test_positions_ingest_request_schema():
    """Test PositionsIngestRequestSchema instantiation and field values."""
    obj = PositionsIngestRequestSchema(
        file_path="/tmp/positions.csv", index_name="positions_index"
    )
    assert obj.file_path == "/tmp/positions.csv"
    assert obj.index_name == "positions_index"
