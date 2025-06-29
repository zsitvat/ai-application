import json
import logging
import os
import tempfile
from urllib.parse import urlparse

import redis
import requests
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain_redis import RedisConfig, RedisVectorStore
from redisvl.schema import IndexSchema

from schemas.graph_schema import Model
from utils.select_model import get_embedding_model

from .default_schema import DEFAULT_INDEX_SCHEMA


class DocumentService:
    """
    Service for document processing and management.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.redis_url = (
            f"redis://{os.getenv('REDIS_USER')}:{os.getenv('REDIS_PASSWORD')}@{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}"
            if os.getenv("REDIS_PASSWORD")
            else f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}"
        )

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

            documents = []

            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        content = item.get("content", "")
                        if not content:
                            self.logger.warning(
                                f"Document {i} in {filename} has no 'content' key or empty content"
                            )
                            content = json.dumps(item, indent=2)

                        metadata = {
                            "filename": filename,
                            "source": file_path,
                            "source_type": "file",
                            "document_index": i,
                        }

                        for key, value in item.items():
                            if key != "content":
                                metadata[key] = str(value) if value is not None else ""

                        doc = Document(page_content=str(content), metadata=metadata)
                        documents.append(doc)
                    else:
                        content = str(item)
                        doc = Document(
                            page_content=content,
                            metadata={
                                "filename": filename,
                                "source": file_path,
                                "source_type": "file",
                                "document_index": i,
                            },
                        )
                        documents.append(doc)
            else:
                if isinstance(data, dict):
                    content = data.get("content", json.dumps(data, indent=2))
                    metadata = {
                        "filename": filename,
                        "source": file_path,
                        "source_type": "file",
                    }

                    for key, value in data.items():
                        if key != "content":
                            metadata[key] = str(value) if value is not None else ""

                    doc = Document(page_content=str(content), metadata=metadata)
                    documents.append(doc)
                else:
                    content = json.dumps(data, indent=2)
                    doc = Document(
                        page_content=content,
                        metadata={
                            "filename": filename,
                            "source": file_path,
                            "source_type": "file",
                        },
                    )
                    documents.append(doc)

            self.logger.debug(
                f"Processed JSON file {filename} into {len(documents)} documents"
            )
            return documents

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in file {filename}: {str(e)}")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return [
                Document(
                    page_content=content,
                    metadata={
                        "filename": filename,
                        "source": file_path,
                        "source_type": "file",
                        "json_parse_error": str(e),
                    },
                )
            ]
        except Exception as e:
            self.logger.error(f"Error processing JSON file {filename}: {str(e)}")
            raise e

    def _process_pdf_file(self, file_path: str) -> list[Document]:
        """Process PDF file."""
        loader = PyPDFLoader(file_path)
        return loader.load()

    def _process_txt_file(self, file_path: str) -> list[Document]:
        """Process text file."""
        loader = TextLoader(file_path)
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
        """Filter document metadata to only include fields defined in the schema.

        Args:
            documents: List of documents to filter
            schema_fields: List of schema field definitions

        Returns:
            List of documents with filtered metadata
        """
        schema_field_names = set()
        for field in schema_fields:
            field_name = field.get("name")
            if field_name and field.get("type") != "vector":
                schema_field_names.add(field_name)

        allowed_fields = schema_field_names

        self.logger.info(f"Schema allows only these fields: {allowed_fields}")

        filtered_documents = []
        dropped_fields_count = 0

        for doc in documents:
            filtered_metadata = {}
            original_metadata_keys = set(doc.metadata.keys())

            for key, value in doc.metadata.items():
                if key in allowed_fields:
                    filtered_metadata[key] = value
                else:
                    dropped_fields_count += 1

            dropped_fields_for_doc = original_metadata_keys - allowed_fields
            if dropped_fields_for_doc:
                self.logger.debug(
                    f"Dropped metadata fields not in schema: {', '.join(dropped_fields_for_doc)}"
                )

            filtered_doc = Document(
                page_content=doc.page_content,
                metadata=filtered_metadata,
            )
            filtered_documents.append(filtered_doc)

        if dropped_fields_count > 0:
            self.logger.info(
                f"Filtered metadata: dropped {dropped_fields_count} field instances not defined in schema"
            )

        return filtered_documents

    def _create_vector_store_and_ingest(
        self,
        documents: list[Document],
        embeddings_model,
        vector_db_index: str,
        chunk_size: int,
        chunk_overlap: int,
        index_schema: list[dict] = None,
    ) -> int:
        """Create vector store and ingest documents.

        Args:
            documents: List of documents to ingest
            embeddings_model: Embeddings model to use
            vector_db_index: Name of the Redis index
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            index_schema: Custom index schema, uses default if None

        Returns:
            int: Number of chunks processed
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        split_documents = text_splitter.split_documents(documents)

        if index_schema is None:
            fields = DEFAULT_INDEX_SCHEMA
        else:
            fields = index_schema

        schema_dict = {
            "index": {"name": vector_db_index, "storage_type": "hash"},
            "fields": fields,
        }
        schema = IndexSchema.from_dict(schema_dict)
        config = RedisConfig.from_schema(schema=schema, redis_url=self.redis_url)

        self.logger.debug(
            f"Filtering metadata according to schema with {len(fields)} fields"
        )
        filtered_documents = self._filter_metadata_by_schema(split_documents, fields)

        try:
            vector_store = RedisVectorStore(embeddings_model, config=config)

            vector_store.add_documents(filtered_documents)

        except Exception as e:
            self.logger.error(
                f"Error adding documents to vector store {vector_db_index}: {str(e)}"
            )
            raise e

        return len(filtered_documents)

    async def ingest_documents(
        self,
        model: Model | None,
        files: list[str],
        vector_db_index: str,
        chunk_size: int,
        chunk_overlap: int,
        index_schema: list[dict] = None,
    ) -> tuple[bool, str, list[str], list[str]]:
        """
        Process and ingest documents into vector database.

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
        try:
            self.logger.info(
                f"Starting document ingestion for {len(files)} files into index '{vector_db_index}'"
            )

            deployment = (
                model.deployment
                if model and model.deployment
                else os.getenv("EMBEDDING_DEPLOYMENT", None)
            )
            if deployment is not None:
                provider = "azure"

            else:
                provider = (
                    model.provider.value
                    if model and model.provider
                    else os.getenv("EMBEDDING_PROVIDER", "openai")
                )

            model_name = (
                model.name
                if model and model.name
                else os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
            )

            self.logger.info(
                f"Embedding configuration - Provider: {provider}, Model: {model_name}, Deployment: {deployment}"
            )

            embeddings_model = await get_embedding_model(
                provider=provider,
                deployment=deployment,
                model=model_name,
            )

            processed_files = []
            failed_files = []

            documents = []
            temp_files_to_cleanup = []

            for file_path in files:
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
                return (
                    False,
                    "No documents were successfully processed",
                    [],
                    failed_files,
                )

            self.logger.info(
                f"Successfully loaded {len(documents)} documents from {len(processed_files)} files"
            )
            self.logger.info(
                f"Starting chunking with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
            )

            chunk_count = self._create_vector_store_and_ingest(
                documents=documents,
                embeddings_model=embeddings_model,
                vector_db_index=vector_db_index,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                index_schema=index_schema,
            )

            self.logger.info(
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

        except Exception as e:
            self.logger.error(f"Error ingesting documents: {str(e)}")
            return (
                False,
                f"Error ingesting documents: {str(e)}",
                [],
                [os.path.basename(f) for f in files],
            )

    async def delete_documents(self, index_name: str) -> tuple[bool, str, int]:
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

    async def get_retriever(
        self,
        index_name: str,
        model: Model,
        index_schema: list[dict] = None,
        search_kwargs: dict | None = None,
    ):
        """
        Get a retriever for the Redis vector database.

        Args:
            index_name (str): Name of the Redis index
            model (Model): Model configuration for embeddings
            index_schema (list[dict], optional): Custom index schema for Redis
            search_kwargs (dict | None): Search parameters

        Returns:
            Retriever: LangChain retriever object
        """
        try:
            provider = (
                model.provider.value
                if model.provider
                else os.getenv("EMBEDDING_PROVIDER", "openai")
            )
            deployment = (
                model.deployment
                if model.deployment
                else os.getenv("EMBEDDING_DEPLOYMENT", None)
            )
            model_name = (
                model.name
                if model.name
                else os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
            )

            embeddings_model = await get_embedding_model(
                provider=provider,
                deployment=deployment,
                model=model_name,
            )

            if index_schema is None:
                fields = DEFAULT_INDEX_SCHEMA
            else:
                fields = index_schema

            schema_dict = {
                "index": {"name": index_name, "storage_type": "hash"},
                "fields": fields,
            }
            schema = IndexSchema.from_dict(schema_dict)
            config = RedisConfig.from_schema(schema, redis_url=self.redis_url)

            vector_store = RedisVectorStore(embeddings_model, config=config)

            if search_kwargs is None:
                search_kwargs = {"k": 5}

            return vector_store.as_retriever(search_kwargs=search_kwargs)

        except Exception as e:
            self.logger.error(f"Error getting retriever: {str(e)}")
            raise e
