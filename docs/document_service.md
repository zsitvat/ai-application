# Document Service

## Overview

The Document Service is responsible for document processing, ingestion and storage in vector database. It supports handling various file formats (PDF, DOCX, TXT, Excel) and storage in Redis Vector Store.

## Main Components

### DocumentService

The `DocumentService` class provides the complete range of document operations, including ingestion, chunking and vectorization.

#### Main Features

- **Document ingestion**: Support for multiple formats (PDF, DOCX, TXT, Excel)
- **Text chunking**: Intelligent chunking based on configuration
- **Vectorization**: Embedding generation and Redis storage
- **Index management**: Dynamic index creation and management
- **File validation**: Secure file processing

## Supported File Formats

- **PDF**: Using PyPDFLoader
- **DOCX**: Using Docx2txtLoader  
- **TXT**: Using TextLoader
- **Excel**: Using UnstructuredExcelLoader

## Usage

### Environment Variables

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_USER=your_username
REDIS_PASSWORD=your_password
```

### Initialization

```python
from src.services.document.document_service import DocumentService

document_service = DocumentService()
```

### Main Methods

#### `ingest_documents(model, files, vector_db_index, chunk_size, chunk_overlap, index_schema)`

Document ingestion and storage in vector database.

**Parameters:**
- `model` (Model|None): Embedding model configuration
- `files` (list[str]): List of files to process
- `vector_db_index` (str): Vector DB index name
- `chunk_size` (int): Size of text chunks
- `chunk_overlap` (int): Overlap between chunks
- `index_schema` (list[dict]): Optional index schema

**Return Value:**
- Tuple: `(success, message, processed_files, failed_files)`

**Example:**
```python
from schemas.model_schema import Model

model = Model(
    model_name="text-embedding-ada-002",
    model_type="embedding"
)

success, message, processed, failed = await document_service.ingest_documents(
    model=model,
    files=["document1.pdf", "document2.docx"],
    vector_db_index="documents_index",
    chunk_size=1000,
    chunk_overlap=200
)
```

#### `get_vector_db_indices()`

List available vector DB indices.

**Return Value:**
- List of index names

#### `delete_vector_db_index(index_name)`

Delete vector DB index.

**Parameters:**
- `index_name` (str): Name of index to delete

**Return Value:**
- Tuple: `(success, message)`

## Text Chunking

### RecursiveCharacterTextSplitter

The service uses LangChain's `RecursiveCharacterTextSplitter`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    is_separator_regex=False,
)
```

### Chunking Strategy

- **Chunk size**: Maximum characters per chunk
- **Overlap**: Overlap to preserve context
- **Recursive**: Intelligent splitting along natural boundaries

## Redis Vector Store Integration

### Configuration

```python
redis_config = RedisConfig(
    index_name=vector_db_index,
    redis_url=self.redis_url,
    metadata_schema=metadata_schema,
    vector_schema=vector_schema
)

vector_store = RedisVectorStore(
    embeddings=embedding_model,
    config=redis_config
)
```

### Index Schema

#### Default Schema

```python
DEFAULT_INDEX_SCHEMA = [
    {
        "name": "content",
        "type": "text"
    },
    {
        "name": "source", 
        "type": "text"
    }
]
```

#### Custom Schema

It is possible to specify a custom index schema according to specific needs.

## File Handling

### Temporary Files

The service automatically handles temporary files:

```python
with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
    # File processing
    pass
os.unlink(temp_file_path)  # Cleanup
```

### Download from URL

Supports downloading files from URLs:

```python
if file_path.startswith(('http://', 'https://')):
    # HTTP download and local save
```

## Error Handling

### File-level Error Handling

- Filtering unsupported formats
- Handling corrupted files
- Network errors (during URL downloads)
- Memory issues with large files

### Result Reporting

```python
return (
    success: bool,
    message: str,
    processed_files: list[str],
    failed_files: list[str]
)
```

## Logging

- **Info level**: Processing progress
- **Debug level**: Chunk size and processing details
- **Error level**: Errors with detailed information
- **Warning level**: Warning messages

## Performance Optimization

### Batch Processing

Processing multiple files simultaneously to increase efficiency.

### Memory Management

- Streaming reading for large files
- Automatic deletion of temporary files
- Chunk-based processing

### Redis Optimization

- Batch insert operations
- Optimized index queries
- Connection reuse

## Security Considerations

### File Validation

- File size checking
- File type validation
- Secure temporary file handling

### URL Security

- URL validation
- Timeout settings
- Secure downloads

## Dependencies

- `langchain`: Document loaders and text splitter
- `langchain_redis`: Redis Vector Store
- `redis`: Redis connection
- `requests`: HTTP file download
- `tempfile`: Temporary file handling
