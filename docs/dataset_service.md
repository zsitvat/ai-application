# Dataset Service

## Overview

The Dataset Service is responsible for managing machine learning datasets using LangSmith integration. This service enables dataset creation, execution, and evaluation.

## Main Components

### DatasetService

The `DatasetService` class provides the complete spectrum of dataset operations through LangSmith integration.

#### Main Features

- **Dataset creation**: Creating new datasets with test data
- **Dataset execution**: Asynchronous and synchronous execution modes
- **Evaluation**: Automatic model evaluation on datasets
- **Background task management**: Handling long-running tasks
- **Error handling**: Detailed error reporting and exception handling

## Usage

### Initialization

```python
from src.services.dataset.dataset_service import DatasetService

dataset_service = DatasetService()
```

### Main Methods

#### `create_dataset(name, description, test_cases)`

Creates a new dataset in LangSmith.

**Parameters:**
- `name` (str): Dataset name
- `description` (str|None): Optional description
- `test_cases` (list[dict]): List of test cases

**Return Value:**
- Dictionary with dataset information

**Exceptions:**
- `DatasetCreationError`: Dataset creation error

**Example:**
```python
test_cases = [
    {"input": "What is the capital?", "expected_output": "Budapest"},
    {"input": "What is 2+2?", "expected_output": "4"}
]

dataset_info = dataset_service.create_dataset(
    name="test_dataset",
    description="Test dataset",
    test_cases=test_cases
)
```

#### `run_dataset_async(dataset_name, run_config)`

Asynchronous dataset execution in background.

**Parameters:**
- `dataset_name` (str): Dataset name
- `run_config` (DatasetRunConfigSchema): Execution configuration

**Return Value:**
- Dictionary with execution ID and status

**Example:**
```python
from schemas.dataset_schema import DatasetRunConfigSchema

run_config = DatasetRunConfigSchema(
    run_name="test_run",
    description="Test execution",
    # additional configuration parameters...
)

result = await dataset_service.run_dataset_async("test_dataset", run_config)
```

#### `run_dataset_sync(dataset_name, run_config)`

Synchronous dataset execution waiting for results.

**Parameters:**
- `dataset_name` (str): Dataset name
- `run_config` (DatasetRunConfigSchema): Execution configuration

**Return Value:**
- Dictionary with execution results

## Configuration Schema

### DatasetRunConfigSchema

```python
class DatasetRunConfigSchema(BaseModel):
    run_name: str
    description: Optional[str]
    evaluation_config: Optional[dict]
    concurrency_level: int = 1
    # additional configuration fields...
```

## Error Handling

### Custom Exceptions

- **DatasetCreationError**: Dataset creation errors
- **DatasetNotFoundError**: Dataset not found
- **DatasetUpdateError**: Dataset update errors
- **DatasetRunError**: Execution errors

### Error Reporting

- Detailed logging of all operations
- Stack trace preservation on errors
- User-friendly error messages

## Background Task Management

The service uses `asyncio.create_task()` to handle background tasks:

```python
self.background_tasks = set()

# Starting background task
task = asyncio.create_task(self._run_dataset_background(...))
self.background_tasks.add(task)
task.add_done_callback(self.background_tasks.discard)
```

## LangSmith Integration

### Client Configuration

```python
from langsmith import Client

self.client = Client()
```

### Supported Operations

- Dataset creation and update
- Execution start and monitoring
- Results retrieval
- Evaluation metrics

## Logging

- **Info level**: Start/end of dataset operations
- **Debug level**: Detailed execution information
- **Error level**: Errors and exceptions
- **Warning level**: Potential issues

## Performance Optimization

- **Asynchronous execution**: For non-blocking operations
- **Concurrency control**: Limiting parallel executions
- **Background tasks**: Avoiding UI blocking
- **Memory management**: Automatic task reference cleanup

## Security Considerations

- **API key management**: LangSmith authentication
- **Input validation**: Input data verification
- **Error information filtering**: Hiding internal details

## Dependencies

- `langsmith`: LangSmith Python SDK
- `langchain.smith.evaluation`: Evaluation tools
- `asyncio`: Asynchronous programming
- `uuid4`: Unique identifier generation
