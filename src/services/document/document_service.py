import json
import os
import tempfile
from urllib.parse import urlparse

import aiofiles
import redis
import requests
from langchain.schema import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain_community.vectorstores.redis import Redis
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from redisvl.schema import IndexSchema

from src.config.app_config import config
from src.schemas.schema import Model
from src.services.logger.logger_service import LoggerService
from src.utils.select_model import get_model

from .default_schema import DEFAULT_INDEX_SCHEMA


class DocumentService:
    """Service for document processing and management.

    This service handles document ingestion, processing, and storage in vector databases.
    It supports various document formats including PDF, DOCX, TXT, and Excel files.

    Attributes:
        logger: Logger instance for debugging and monitoring
        redis_url: Connection URL for Redis database
    """

    def __init__(self) -> None:
        """Initialize the DocumentService with logger and Redis configuration."""
        self.logger = LoggerService().setup_logger()
        self.redis_url = config.redis.url

    async def ingest_documents(
        self,
        vector_db_index: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        files: list[str] | None = None,
        index_schema: list[dict[str, any]] | None = None,
        json_data: list[dict[str, any]] | None = None,
        model: Model | None = None,
    ) -> tuple[bool, str, list[str], list[str]]:
        """Process and ingest documents into vector database.

        This method handles document ingestion from either files or JSON data,
        processes them into chunks, and stores them in a vector database.

        Args:
            vector_db_index: Vector database index name
            chunk_size: Size of text chunks for processing (default: 5000)
            chunk_overlap: Overlap between chunks (default: 250)
            files: List of file paths to process (optional)
            index_schema: Custom index schema for Redis (optional)
            json_data: Direct JSON data for ingestion (optional)
            model: Model configuration for embeddings (optional)

        Returns:
            Tuple containing:
                - success: Boolean indicating if operation succeeded
                - message: Status message
                - processed_files: List of successfully processed files
                - failed_files: List of files that failed processing

        Raises:
            Exception: When document processing or ingestion fails
        """
        # Use config defaults if not provided
        chunk_size = chunk_size or config.document.chunk_size
        chunk_overlap = chunk_overlap or config.document.chunk_overlap

        self.logger.info(
            f"[DocumentService|ingest_documents] started (vector_db_index={vector_db_index})"
        )
        try:
            if json_data is not None and len(json_data) > 0:
                return self.ingest_documents_from_json(
                    model=model,
                    vector_db_index=vector_db_index,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    index_schema=index_schema,
                    json_data=json_data,
                )

            else:
                return await self.ingest_documents_from_files(
                    model=model,
                    files=files or [],
                    vector_db_index=vector_db_index,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    index_schema=index_schema,
                )

        except Exception as e:
            self.logger.error(f"Error ingesting documents: {str(e)}")
            self.logger.info(
                f"[DocumentService|ingest_documents] finished (vector_db_index={vector_db_index})"
            )
            return (
                False,
                f"Error ingesting documents: {str(e)}",
                [],
                [os.path.basename(f) for f in files] if files else [],
            )

    def ingest_documents_from_json(
        self,
        vector_db_index: str,
        chunk_size: int = 5000,
        chunk_overlap: int = 250,
        index_schema: list[dict] | None = None,
        json_data: list[dict] | None = None,
        model: Model | None = None,
    ) -> tuple[bool, str, list[str], list[str]]:
        self.logger.info(
            f"Starting document ingestion from direct JSON data into index '{vector_db_index}'"
        )

        embeddings_model = get_model(
            provider=(
                model.provider
                if model and model.provider
                else os.getenv("EMBEDDING_PROVIDER", "")
            ),
            deployment=(
                model.deployment
                if model and model.deployment
                else os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME", "")
            ),
            model=(
                model.name if model and model.name else os.getenv("EMBEDDING_MODEL", "")
            ),
            type="embedding",
        )

        processed_files = ["json_data"]
        failed_files = []
        documents = self._process_json_data(json_data if json_data is not None else [])

        chunk_count = self._create_vector_store_and_ingest(
            documents=documents,
            embeddings_model=embeddings_model,
            vector_db_index=vector_db_index,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            index_schema=index_schema,
        )

        self.logger.debug(
            f"Successfully ingested {chunk_count} chunks from direct JSON data into {vector_db_index}"
        )

        return (
            True,
            f"Successfully processed direct JSON data into {chunk_count} chunks",
            processed_files,
            failed_files,
        )

    async def ingest_documents_from_files(
        self,
        vector_db_index: str,
        chunk_size: int = 5000,
        chunk_overlap: int = 250,
        files: list[str] = [],
        index_schema: list[dict] | None = None,
        model: Model | None = None,
    ) -> tuple[bool, str, list[str], list[str]]:
        """Process and ingest documents from files into vector database.
        Args:
            model (Model): Model configuration for embeddings
            files (list[str]): List of file paths
            vector_db_index (str): Vector DB index name
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
            index_schema (list[dict], optional): Custom index schema for Redis
        Returns:
            tuple[bool, str, list[str], list[str]]: (success, message, processed_files, failed_files)
        """

        self.logger.info(
            f"Starting document ingestion for {len(files) if files is not None else 0} files into index '{vector_db_index}'"
        )

        embeddings_model = get_model(
            provider=(
                model.provider
                if model and model.provider
                else os.getenv("EMBEDDING_PROVIDER", "")
            ),
            deployment=(
                model.deployment
                if model and model.deployment
                else os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME", "")
            ),
            model=(
                model.name if model and model.name else os.getenv("EMBEDDING_MODEL", "")
            ),
            type="embedding",
        )

        processed_files = []
        failed_files = []
        documents = []
        temp_files_to_cleanup = []

        expanded_files = []
        for file_path in files:
            if os.path.isdir(file_path):
                for entry in os.listdir(file_path):
                    full_path = os.path.join(file_path, entry)
                    if os.path.isfile(full_path):
                        expanded_files.append(full_path)
            else:
                expanded_files.append(file_path)

        for file_path in expanded_files:

            try:
                current_file_path = file_path.strip()
                if self._is_url(file_path):
                    self.logger.debug(f"Downloading file from URL: {file_path}")
                    current_file_path = self._download_file_from_url(file_path)
                    temp_files_to_cleanup.append(current_file_path)
                if not os.path.exists(current_file_path):
                    raise FileNotFoundError(f"File not found: {current_file_path}")

                filename = (
                    os.path.basename(file_path)
                    if self._is_url(file_path)
                    else os.path.basename(current_file_path)
                )
                file_extension = (
                    filename.lower().split(".")[-1] if "." in filename else ""
                )

                if file_extension == "json":
                    file_documents = self._process_json_file(
                        current_file_path, filename
                    )
                elif file_extension == "pdf":
                    file_documents = self._process_pdf_file(current_file_path)
                elif file_extension == "txt":
                    file_documents = self._process_txt_file(current_file_path)
                elif file_extension == "docx":
                    file_documents = self._process_docx_file(current_file_path)
                elif file_extension == "xlsx":
                    file_documents = self._process_xlsx_file(current_file_path)
                else:
                    raise ValueError(
                        f"Unsupported file type: {file_extension}. Supported types: txt, json, docx, pdf, xlsx"
                    )

                if file_extension != "json":
                    for doc in file_documents:
                        doc.metadata["filename"] = filename
                        doc.metadata["source"] = file_path

                documents.extend(file_documents)
                processed_files.append(filename)

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {str(e)}")
                failed_files.append(
                    os.path.basename(file_path) if file_path else "unknown"
                )

        if not documents:
            self.logger.info(
                f"[DocumentService|ingest_documents_from_files] finished (vector_db_index={vector_db_index})"
            )

            return (
                False,
                "No documents were successfully processed",
                [],
                failed_files,
            )

        self.logger.info(
            f"Successfully loaded {len(documents)} documents from {len(processed_files)} files"
        )

        chunk_count = self._create_vector_store_and_ingest(
            documents=documents,
            embeddings_model=embeddings_model,
            vector_db_index=vector_db_index,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            index_schema=index_schema,
        )

        self.logger.debug(
            f"Vector store ingestion completed: {chunk_count} chunks indexed"
        )

        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as cleanup_error:
                self.logger.warning(
                    f"Failed to cleanup temp file {temp_file}: {str(cleanup_error)}"
                )

        self.logger.debug(
            f"Successfully ingested {chunk_count} chunks from {len(processed_files)} files into {vector_db_index}"
        )

        return (
            True,
            f"Successfully processed {len(processed_files)} files into {chunk_count} chunks",
            processed_files,
            failed_files,
        )

    def _process_json_data(self, data: list[dict]) -> list[Document]:
        """
        Process a list of dicts as documents (same as _process_json_file but from memory).
        """

        documents = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                content = item.get("content", json.dumps(item, ensure_ascii=False))
                metadata = {k: str(v) for k, v in item.items() if k != "content"}
                metadata["document_index"] = str(i)
                doc = Document(page_content=str(content), metadata=metadata)
                documents.append(doc)
            else:
                doc = Document(page_content=str(item), metadata={"document_index": i})
                documents.append(doc)
        return documents

    def delete_documents(self, index_name: str) -> tuple[bool, str, int]:
        """
        Delete documents from vector database.

        Args:
            index_name (str): Vector DB index name

        Returns:
            tuple[bool, str, int]: (success, message, deleted_count)
        """
        try:
            r = redis.from_url(self.redis_url)

            try:
                index_info = r.execute_command("FT.INFO", index_name)
                doc_count = 0
                for i, item in enumerate(index_info):
                    if isinstance(item, bytes) and item.decode() == "num_docs":
                        doc_count = int(index_info[i + 1])
                        break

                r.execute_command("FT.DROPINDEX", index_name, "DD")

                message = f"Successfully deleted index {index_name} with {doc_count} documents"
                self.logger.info(message)
                return True, message, doc_count

            except redis.ResponseError as e:
                if "Unknown index name" in str(e):
                    message = f"Index {index_name} not found"
                    self.logger.warning(message)
                    return False, message, 0
                raise e

        except Exception as e:
            message = f"Error deleting documents: {str(e)}"
            self.logger.error(message)
            return False, message, 0

    def get_retriever(
        self,
        index_name: str,
        model: Model | None = None,
        index_schema: list[dict] | None = None,
        search_kwargs: dict | None = None,
        search_type: str = "similarity",
    ):
        """
        Get a retriever for the Redis vector database.

        Args:
            index_name (str): Name of the Redis index
            model (Model | None): Model configuration for embeddings
            index_schema (list[dict] | None): Custom index schema for Redis
            search_kwargs (dict | None): Search parameters

        Returns:
            Retriever: LangChain retriever object
        """
        try:

            embeddings_model = get_model(
                provider=(
                    model.provider
                    if model and model.provider
                    else os.getenv("EMBEDDING_PROVIDER", "")
                ),
                deployment=(
                    model.deployment
                    if model and model.deployment
                    else os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME", "")
                ),
                model=(
                    model.name
                    if model and model.name
                    else os.getenv("EMBEDDING_MODEL", "")
                ),
                type="embedding",
            )

            if search_kwargs is None:
                search_kwargs = {"k": 10}

            index_schema = index_schema or [{"name": "content", "type": "text"}]

            vector_store = Redis.from_existing_index(
                embedding=embeddings_model,
                index_name=index_name,
                schema={"fields": index_schema},
                redis_url=self.redis_url,
            )

            return vector_store.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )

        except Exception as e:
            self.logger.error(f"Error getting retriever: {str(e)}")
            raise e

    def _is_url(self, string: str) -> bool:
        """Check if a string is a valid URL."""
        try:
            result = urlparse(string)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _download_file_from_url(self, url: str) -> str:
        """Download a file from URL and return temporary file path."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(response.content)
                return tmp_file.name

        except Exception as e:
            self.logger.error(f"Error downloading file from URL {url}: {str(e)}")
            raise e

    def _process_json_file(self, file_path: str, filename: str) -> list[Document]:
        """Process JSON file containing list of objects as documents.

        Expects JSON format: [{"content": "text", "key1": "value1", "key2": "value2"}, ...]
        The 'content' key becomes the document text, all other keys become metadata.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                documents = self._process_json_list(data, filename, file_path)
            elif isinstance(data, dict):
                documents = self._process_json_dict(data, filename, file_path)
            else:
                documents = [self._process_json_other(data, filename, file_path)]

            self.logger.info(
                f"Processed JSON file {filename}: {len(documents)} documents"
            )
            return documents

        except Exception as e:
            self.logger.error(f"Error processing JSON file {filename}: {str(e)}")
            raise e

    def _process_json_list(self, data_list, filename, file_path):
        documents = []
        for i, item in enumerate(data_list):
            if isinstance(item, dict):
                doc = self._process_json_list_item_dict(item, filename, file_path, i)
            else:
                doc = Document(
                    page_content=str(item),
                    metadata={
                        "filename": filename,
                        "source": file_path,
                        "source_type": "file",
                        "document_index": i,
                    },
                )
            documents.append(doc)
        return documents

    def _process_json_list_item_dict(self, item, filename, file_path, i):
        if "content" not in item:
            item = dict(item)
            item["content"] = json.dumps(item, ensure_ascii=False)
        content = item.get("content", "")
        metadata = {
            "filename": filename,
            "source": file_path,
            "source_type": "file",
            "document_index": i,
        }
        for key, value in item.items():
            if key != "content":
                metadata[key] = str(value) if value is not None else ""
        return Document(page_content=str(content), metadata=metadata)

    def _process_json_dict(self, data_dict, filename, file_path):
        documents = []
        for key, value in data_dict.items():
            if isinstance(value, dict):
                doc = self._process_json_dict_item_dict(value, filename, file_path, key)
            else:
                doc = Document(
                    page_content=str(value),
                    metadata={
                        "filename": filename,
                        "source": file_path,
                        "source_type": "file",
                        "document_key": key,
                    },
                )
            documents.append(doc)
        return documents

    def _process_json_dict_item_dict(self, value, filename, file_path, key):
        content = value.get("content", json.dumps(value, indent=2))
        metadata = {
            "filename": filename,
            "source": file_path,
            "source_type": "file",
            "document_key": key,
        }
        for k, v in value.items():
            if k != "content":
                metadata[k] = str(v) if v is not None else ""
        return Document(page_content=str(content), metadata=metadata)

    def _process_json_other(self, data, filename, file_path):
        return Document(
            page_content=json.dumps(data, indent=2),
            metadata={
                "filename": filename,
                "source": file_path,
                "source_type": "file",
            },
        )

    def _process_pdf_file(self, file_path: str) -> list[Document]:
        """Process PDF file."""
        loader = PyPDFLoader(file_path)
        return loader.load()

    def _process_txt_file(self, file_path: str) -> list[Document]:
        """Process text file."""
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    def _process_docx_file(self, file_path: str) -> list[Document]:
        """Process DOCX file."""
        loader = Docx2txtLoader(file_path)
        return loader.load()

    def _process_xlsx_file(self, file_path: str) -> list[Document]:
        """Process Excel file."""
        loader = UnstructuredExcelLoader(file_path)
        return loader.load()

    def _filter_metadata_by_schema(
        self, documents: list[Document], schema_fields: list[dict]
    ) -> list[Document]:
        """Filter document metadata according to the index schema.

        Args:
            documents (list[Document]): List of documents to filter
            schema_fields (list[dict]): Index schema fields definition

        Returns:
            list[Document]: Documents with filtered metadata
        """
        if not schema_fields:
            return documents

        field_names = set()
        for field in schema_fields:
            field_names.add(field.get("name", ""))

        filtered_documents = []
        for doc in documents:
            if doc.metadata:
                filtered_metadata = {}
                for key, value in doc.metadata.items():
                    if key in field_names:
                        filtered_metadata[key] = value

                filtered_doc = Document(
                    page_content=doc.page_content, metadata=filtered_metadata
                )
                filtered_documents.append(filtered_doc)
            else:
                filtered_documents.append(doc)

        self.logger.debug(
            f"Filtered metadata for {len(documents)} documents according to schema"
        )
        return filtered_documents

    def _create_vector_store_and_ingest(
        self,
        documents: list[Document],
        embeddings_model,
        vector_db_index: str,
        chunk_size: int,
        chunk_overlap: int,
        index_schema: list[dict] | None = None,
    ) -> int:
        """Create vector store and ingest documents.

        Args:
            documents (list[Document]): Documents to ingest
            embeddings_model: Embedding model instance
            vector_db_index (str): Vector database index name
            chunk_size (int): Text chunk size
            chunk_overlap (int): Chunk overlap size
            index_schema (list[dict], optional): Index schema definition

        Returns:
            int: Number of chunks created and ingested
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            )

            chunks = text_splitter.split_documents(documents)

            if index_schema:
                chunks = self._filter_metadata_by_schema(chunks, index_schema)
                fields = index_schema
            else:
                fields = DEFAULT_INDEX_SCHEMA

            self.logger.info(f"Created {len(chunks)} chunks from documents")

            schema_dict = {
                "index": {"name": vector_db_index, "storage_type": "hash"},
                "fields": fields,
            }

            schema = IndexSchema.from_dict(schema_dict)
            config = RedisConfig.from_schema(schema, redis_url=self.redis_url)

            vector_store = RedisVectorStore(embeddings_model, config=config)

            vector_store.add_documents(chunks)

            self.logger.info(
                f"Successfully ingested {len(chunks)} chunks into Redis vector store"
            )

            return len(chunks)

        except Exception as e:
            self.logger.error(f"Error creating vector store and ingesting: {str(e)}")
            raise e

    async def ingest_positions(self, file_path: str, index_name: str):
        """
        Ingest positions from a file, flatten labels, and store in vector DB using index schema.
        Args:
            file_path (str): Path to the positions file (JSON)
            index_name (str): Name of the Redis index
        Returns:
            tuple: (success, message, processed_files, failed_files)
        """
        try:
            index_schema = await self._load_positions_index_schema()
            if index_schema is None:
                error_msg = "Missing POSITIONS_INDEX_SCHEMA_PATH environment variable."
                self.logger.error(error_msg)
                return False, error_msg, [], []
            if not file_path:
                error_msg = "Missing file_path argument for positions ingestion."
                self.logger.error(error_msg)
                return False, error_msg, [], []

            all_positions = await self._load_and_flatten_positions(file_path)

            success, message, processed_files, failed_files = (
                await self.ingest_documents(
                    model=None,
                    json_data=all_positions,
                    vector_db_index=index_name,
                    chunk_size=5000,
                    chunk_overlap=200,
                    index_schema=index_schema,
                )
            )

            return success, message, processed_files, failed_files

        except Exception as ex:
            self.logger.error(f"Error in positions ingestion: {str(ex)}")
            return False, f"Error processing positions: {str(ex)}", [], []

    async def _load_positions_index_schema(self):
        schema_path = os.getenv("POSITIONS_INDEX_SCHEMA_PATH")
        if not schema_path:
            return None
        async with aiofiles.open(schema_path, "r", encoding="utf-8") as f:
            return json.loads(await f.read())

    async def _load_and_flatten_positions(self, file_path: str):
        all_positions = []
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
            positions = json.loads(content)
            if isinstance(positions, list):
                for pos in positions:
                    all_positions.append(self._flatten_labels(pos))
            else:
                self.logger.error(
                    "Positions data is not a list. Skipping label flattening."
                )
                all_positions = positions
        return all_positions

    def _flatten_labels(self, pos):
        if isinstance(pos, dict):
            labels = pos.get("labels", {})
            if isinstance(labels, dict):
                for key, value in labels.items():
                    pos[f"labels_{key}"] = value
            pos.pop("labels", None)
        return pos
