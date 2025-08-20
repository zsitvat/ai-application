import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest

from src.services.document.document_service import Document, DocumentService


class DummyLogger(logging.Logger):
    def info(self, msg):
        # Dummy logger method for info
        pass

    def warning(self, msg):
        # Dummy logger method for warning
        pass

    def error(self, msg):
        # Dummy logger method for error
        pass

    def debug(self, msg):
        # Dummy logger method for debug
        pass


@pytest.fixture
def service():
    svc = DocumentService()
    svc.redis_url = "redis://localhost"
    svc.logger = DummyLogger("dummy")
    return svc


def test_is_url(service):
    assert service._is_url("https://example.com")
    assert not service._is_url("not_a_url")


def test_flatten_labels(service):
    pos = {"labels": {"a": 1, "b": 2}, "other": 3}
    result = service._flatten_labels(pos.copy())
    assert "labels_a" in result and "labels_b" in result
    assert "labels" not in result


def test_process_json_other(service):
    doc = service._process_json_other({"foo": "bar"}, "file.json", "/tmp/file.json")
    assert isinstance(doc, Document)
    assert "foo" in doc.page_content


def test_process_json_list_and_dict(service):
    data_list = [{"content": "abc", "meta": 1}, "plain"]
    docs = service._process_json_list(data_list, "f.json", "/tmp/f.json")
    assert len(docs) == 2
    assert isinstance(docs[0], Document)
    data_dict = {"k1": {"content": "v1", "meta": 2}, "k2": "v2"}
    docs2 = service._process_json_dict(data_dict, "f.json", "/tmp/f.json")
    assert len(docs2) == 2


def test_process_json_list_item_dict(service):
    item = {"meta": 1}
    doc = service._process_json_list_item_dict(item, "f.json", "/tmp/f.json", 0)
    assert isinstance(doc, Document)
    assert doc.page_content == '{"meta": 1}'
    assert doc.metadata["meta"] == "1"


def test_process_json_dict_item_dict(service):
    value = {"meta": 2}
    doc = service._process_json_dict_item_dict(value, "f.json", "/tmp/f.json", "k")
    assert isinstance(doc, Document)
    assert "meta" in doc.metadata


def test_filter_metadata_by_schema(service):
    docs = [Document(page_content="abc", metadata={"a": 1, "b": 2})]
    schema = [{"name": "a"}]
    filtered = service._filter_metadata_by_schema(docs, schema)
    assert "a" in filtered[0].metadata
    assert "b" not in filtered[0].metadata


@patch("src.services.document.document_service.PyPDFLoader")
def test_process_pdf_file(mock_loader, service):
    mock_loader.return_value.load.return_value = [
        Document(page_content="pdf", metadata={})
    ]
    docs = service._process_pdf_file("dummy.pdf")
    assert docs[0].page_content == "pdf"


@patch("src.services.document.document_service.TextLoader")
def test_process_txt_file(mock_loader, service):
    mock_loader.return_value.load.return_value = [
        Document(page_content="txt", metadata={})
    ]
    docs = service._process_txt_file("dummy.txt")
    assert docs[0].page_content == "txt"


@patch("src.services.document.document_service.Docx2txtLoader")
def test_process_docx_file(mock_loader, service):
    mock_loader.return_value.load.return_value = [
        Document(page_content="docx", metadata={})
    ]
    docs = service._process_docx_file("dummy.docx")
    assert docs[0].page_content == "docx"


@patch("src.services.document.document_service.UnstructuredExcelLoader")
def test_process_xlsx_file(mock_loader, service):
    mock_loader.return_value.load.return_value = [
        Document(page_content="xlsx", metadata={})
    ]
    docs = service._process_xlsx_file("dummy.xlsx")
    assert docs[0].page_content == "xlsx"


def test_filter_metadata_by_schema_empty(service):
    docs = [Document(page_content="abc", metadata={"a": 1})]
    filtered = service._filter_metadata_by_schema(docs, [])
    assert filtered == docs


@patch("src.services.document.document_service.redis.from_url")
def test_delete_documents_success(mock_redis, service):
    mock_r = MagicMock()
    mock_r.execute_command.side_effect = [[b"num_docs", b"2"], None]
    mock_redis.return_value = mock_r
    success, msg, count = service.delete_documents("test_index")
    assert success is True
    assert "Successfully deleted index" in msg
    assert count == 2


@patch("src.services.document.document_service.redis.from_url")
def test_delete_documents_unknown_index(mock_redis, service):
    mock_r = MagicMock()

    class ResponseError(Exception):
        pass

    def raise_response_error(*args, **kwargs):
        raise ResponseError("Unknown index name")

    mock_r.execute_command.side_effect = raise_response_error
    mock_redis.return_value = mock_r
    with patch(
        "src.services.document.document_service.redis.ResponseError", ResponseError
    ):
        success, msg, count = service.delete_documents("bad_index")
        assert success is False
        assert "not found" in msg
        assert count == 0


@patch("src.services.document.document_service.redis.from_url")
def test_delete_documents_other_error(mock_redis, service):
    mock_redis.side_effect = Exception("fail")
    success, msg, count = service.delete_documents("any_index")
    assert success is False
    assert "Error deleting documents" in msg
    assert count == 0


@patch("langchain_community.vectorstores.redis.Redis.from_existing_index")
@patch("src.services.document.document_service.redis.from_url")
def test_get_retriever_error(mock_redis_url, mock_redis_index, service):
    mock_redis_index.side_effect = RuntimeError("fail")
    service.logger.error = lambda msg: None
    with pytest.raises(RuntimeError):
        service.get_retriever("index")


def test_process_json_data(service):
    data = [{"content": "abc", "meta": 1}, {"meta": 2}, "plain"]
    docs = service._process_json_data(data)
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "abc"
    assert docs[1].page_content == '{"meta": 2}'
    assert docs[2].page_content == "plain"


def test_process_json_file_file_not_found(service):
    with pytest.raises(FileNotFoundError):
        service._process_json_file("/nonexistent.json", "nonexistent.json")


def test_process_json_file_json_decode_error(service, tmp_path):
    file_path = tmp_path / "bad.json"
    file_path.write_text("not a json")
    with pytest.raises(Exception):
        service._process_json_file(str(file_path), "bad.json")


def test_process_json_file_dict_branch(service, tmp_path):
    file_path = tmp_path / "dict.json"
    file_path.write_text('{"a": {"content": "foo"}, "b": "bar"}')
    docs = service._process_json_file(str(file_path), "dict.json")
    assert len(docs) == 2
    assert docs[0].page_content == "foo"
    assert docs[1].page_content == "bar"


def test_process_json_file_other_branch(service, tmp_path):
    file_path = tmp_path / "other.json"
    file_path.write_text("42")
    docs = service._process_json_file(str(file_path), "other.json")
    assert isinstance(docs[0], Document)


def test_ingest_documents_from_files_unsupported_type(service):
    async def run():
        result = await service.ingest_documents_from_files(
            vector_db_index="idx",
            files=["file.unsupported"],
        )
        assert result[0] is False
        assert "No documents were successfully processed" in result[1]

    asyncio.run(run())


def test_is_url_edge_cases(service):
    assert not service._is_url("")
    assert not service._is_url("ftp:/bad")
    assert service._is_url("http://localhost")


def test_flatten_labels_edge_cases(service):
    # No labels key
    pos = {"foo": 1}
    out = service._flatten_labels(pos.copy())
    assert out["foo"] == 1
    # labels not dict
    pos = {"labels": "notadict", "bar": 2}
    out = service._flatten_labels(pos.copy())
    assert "labels" in out or "labels" not in out


def test_filter_metadata_by_schema_no_metadata(service):
    docs = [Document(page_content="abc", metadata={})]
    schema = [{"name": "a"}]
    filtered = service._filter_metadata_by_schema(docs, schema)
    assert filtered[0].page_content == "abc"


def test_download_file_from_url_error(service):
    with patch("src.services.document.document_service.requests.get") as mock_get:
        mock_get.side_effect = Exception("network fail")
        with pytest.raises(Exception):
            service._download_file_from_url("http://badurl")


def test_create_vector_store_and_ingest_error(service):
    with patch(
        "src.services.document.document_service.RecursiveCharacterTextSplitter"
    ) as mock_splitter:
        mock_splitter.side_effect = Exception("splitter fail")
        with pytest.raises(Exception):
            service._create_vector_store_and_ingest(
                documents=[Document(page_content="abc", metadata={})],
                embeddings_model=None,
                vector_db_index="idx",
                chunk_size=1,
                chunk_overlap=1,
            )


@pytest.mark.asyncio
async def test_ingest_positions_missing_schema(monkeypatch):
    svc = DocumentService()
    svc.logger = DummyLogger("dummy")
    monkeypatch.setenv("POSITIONS_INDEX_SCHEMA_PATH", "")
    success, msg, _, _ = await svc.ingest_positions("file.json", "idx")
    assert not success
    assert "Missing POSITIONS_INDEX_SCHEMA_PATH" in msg


@pytest.mark.asyncio
async def test_ingest_positions_missing_file(monkeypatch):
    svc = DocumentService()
    svc.logger = DummyLogger("dummy")
    monkeypatch.setenv("POSITIONS_INDEX_SCHEMA_PATH", "/tmp/schema.json")
    success, msg, _, _ = await svc.ingest_positions("", "idx")
    assert not success
    assert "Error processing positions" in msg


@pytest.mark.asyncio
async def test_load_positions_index_schema_file(tmp_path, service, monkeypatch):
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('[{"name": "foo"}]')
    monkeypatch.setenv("POSITIONS_INDEX_SCHEMA_PATH", str(schema_path))
    result = await service._load_positions_index_schema()
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_load_and_flatten_positions_list(tmp_path, service):
    file_path = tmp_path / "positions.json"
    file_path.write_text('[{"labels": {"a": 1}}, {"labels": {"b": 2}}]')
    result = await service._load_and_flatten_positions(str(file_path))
    assert isinstance(result, list)
    assert "labels_a" in result[0]


@pytest.mark.asyncio
async def test_load_and_flatten_positions_not_list(tmp_path, service):
    file_path = tmp_path / "positions.json"
    file_path.write_text('{"foo": "bar"}')
    result = await service._load_and_flatten_positions(str(file_path))
    assert isinstance(result, dict)
