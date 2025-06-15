import logging
import os
import redis
import requests
import tempfile
import json
from urllib.parse import urlparse
from langchain_redis import RedisConfig, RedisVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
)
from langchain.schema import Document

from schemas.graph_schema import Model
from utils.model_selector import get_embedding_model
from .default_schema import DEFAULT_INDEX_SCHEMA


class DocumentService:
    """
    Service for document processing and management.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.redis_url = os.getenv("REDIS_URL")

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
            index_schema = DEFAULT_INDEX_SCHEMA

        config = RedisConfig(
            index_name=vector_db_index,
            redis_url=self.redis_url,
            index_schema=index_schema,
        )

        vector_store = RedisVectorStore(embeddings_model, config=config)
        vector_store.add_documents(split_documents)

        return len(split_documents)

    async def ingest_documents(
        self,
        model: Model,
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

            embeddings_model = await get_embedding_model(
                provider=model.provider.value,
                deployment=model.deployment,
                model=model.name,
            )

            processed_files = []
            failed_files = []

            documents = []
            temp_files_to_cleanup = []

            for file_path in files:
                try:
                    current_file_path = file_path

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
                            doc.metadata["source_type"] = (
                                "url" if self._is_url(file_path) else "file"
                            )

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

            chunk_count = self._create_vector_store_and_ingest(
                documents=documents,
                embeddings_model=embeddings_model,
                vector_db_index=vector_db_index,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                index_schema=index_schema,
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
            embeddings_model = await get_embedding_model(
                provider=model.provider.value,
                deployment=model.deployment,
                model=model.name,
            )

            if index_schema is None:
                index_schema = DEFAULT_INDEX_SCHEMA

            config = RedisConfig(
                index_name=index_name,
                redis_url=self.redis_url,
                index_schema=index_schema,
            )

            vector_store = RedisVectorStore(embeddings_model, config=config)

            if search_kwargs is None:
                search_kwargs = {"k": 5}

            return vector_store.as_retriever(search_kwargs=search_kwargs)

        except Exception as e:
            self.logger.error(f"Error getting retriever: {str(e)}")
            raise e
