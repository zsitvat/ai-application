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

### Inicializálás

```python
from src.services.document.document_service import DocumentService

document_service = DocumentService()
```

### Főbb metódusok

#### `ingest_documents(model, files, vector_db_index, chunk_size, chunk_overlap, index_schema)`

Dokumentumok beolvasása és vektoradatbázisba mentése.

**Paraméterek:**
- `model` (Model|None): Embedding modell konfiguráció
- `files` (list[str]): Feldolgozandó fájlok listája
- `vector_db_index` (str): Vector DB index neve
- `chunk_size` (int): Szövegdarabok mérete
- `chunk_overlap` (int): Átfedés mértéke chunkok között
- `index_schema` (list[dict]): Opcionális index séma

**Visszatérési érték:**
- Tuple: `(success, message, processed_files, failed_files)`

**Példa:**
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

Elérhető vector DB indexek listázása.

**Visszatérési érték:**
- Lista az index nevekkel

#### `delete_vector_db_index(index_name)`

Vector DB index törlése.

**Paraméterek:**
- `index_name` (str): Törlendő index neve

**Visszatérési érték:**
- Tuple: `(success, message)`

## Szöveg darabolás

### RecursiveCharacterTextSplitter

A service a LangChain `RecursiveCharacterTextSplitter`-t használja:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    is_separator_regex=False,
)
```

### Darabolási stratégia

- **Chunk size**: Maximális karakterszám per chunk
- **Overlap**: Átfedés biztosítása a kontextus megőrzéséhez
- **Rekurzív**: Intelligens felosztás természetes határok mentén

## Redis Vector Store integráció

### Konfiguráció

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

### Index séma

#### Alapértelmezett séma

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

#### Egyedi séma

Lehetőség van egyedi index séma megadására a specifikus igények szerint.

## Fájl kezelés

### Ideiglenes fájlok

A service automatikusan kezeli az ideiglenes fájlokat:

```python
with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
    # Fájl feldolgozás
    pass
os.unlink(temp_file_path)  # Cleanup
```

### URL-ből letöltés

Támogatja a fájlok URL-ből való letöltését:

```python
if file_path.startswith(('http://', 'https://')):
    # HTTP letöltés és helyi mentés
```

## Hibakezelés

### Fájl szintű hibakezelés

- Nem támogatott formátumok kiszűrése
- Sérült fájlok kezelése
- Hálózati hibák (URL letöltésnél)
- Memória problémák nagyobb fájloknál

### Eredményjelentés

```python
return (
    success: bool,
    message: str,
    processed_files: list[str],
    failed_files: list[str]
)
```

## Naplózás

- **Info szint**: Feldolgozási előrehaladás
- **Debug szint**: Chunk méret és feldolgozási részletek
- **Error szint**: Hibák részletes információkkal
- **Warning szint**: Figyelmeztető üzenetek

## Teljesítmény optimalizáció

### Batch feldolgozás

Több fájl egyszerre történő feldolgozása hatékonyság növelése érdekében.

### Memória kezelés

- Streaming olvasás nagyobb fájlokhoz
- Ideiglenes fájlok automatikus törlése
- Chunk-alapú feldolgozás

### Redis optimalizáció

- Batch insert műveletek
- Index optimalizált lekérdezések
- Kapcsolat újrafelhasználás

## Biztonsági szempontok

### Fájl validáció

- Fájl méret ellenőrzés
- Fájltípus validáció
- Biztonságos ideiglenes fájl kezelés

### URL biztonság

- URL validáció
- Timeout beállítások
- Biztonságos letöltés

## Függőségek

- `langchain`: Document loaderek és text splitter
- `langchain_redis`: Redis Vector Store
- `redis`: Redis kapcsolat
- `requests`: HTTP fájl letöltés
- `tempfile`: Ideiglenes fájl kezelés
