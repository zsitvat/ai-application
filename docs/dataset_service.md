# Dataset Service

The Dataset Service provides comprehensive dataset management functionality using LangSmith for creating, managing, and running test datasets against your multi-agent AI system.

## Features

- **Create Datasets**: Create new test datasets with multiple test cases
- **Retrieve Datasets**: Get existing datasets by name
- **Update Datasets**: Modify dataset descriptions and test cases
- **Run Datasets**: Execute test cases against your AI system and collect results

## Prerequisites

1. **LangSmith API Key**: Set your LangSmith API key as an environment variable:
   ```bash
   export LANGSMITH_API_KEY="your_langsmith_api_key"
   ```

2. **LangSmith Project**: Configure your LangSmith project in the environment:
   ```bash
   export LANGSMITH_PROJECT="your_project_name"
   ```

## API Endpoints

### Create Dataset
- **Endpoint**: `POST /api/dataset`
- **Description**: Creates a new test dataset
- **Request Body**:
  ```json
  {
    "dataset_name": "my_test_dataset",
    "description": "Dataset for testing AI responses",
    "test_cases": [
      {
        "input": "What is AI?",
        "expected_output": "Artificial Intelligence is...",
        "metadata": {"category": "AI basics"}
      },
      {
        "inputs": {"question": "How does ML work?"},
        "outputs": {"answer": "Machine Learning works by..."},
        "metadata": {"category": "ML"}
      }
    ]
  }
  ```

### Get Dataset
- **Endpoint**: `GET /api/dataset/{dataset_name}`
- **Description**: Retrieves a dataset by name

### Update Dataset
- **Endpoint**: `PATCH /api/dataset/{dataset_name}`
- **Description**: Updates an existing dataset
- **Request Body**:
  ```json
  {
    "description": "Updated description",
    "test_cases": [...]
  }
  ```

### Run Dataset
- **Endpoint**: `POST /api/dataset/{dataset_name}/run`
- **Description**: Runs test cases against the AI system
- **Request Body**:
  ```json
  {
    "config": {
      "model": "gpt-4o-mini",
      "temperature": 0.1,
      "max_tokens": 150
    }
  }
  ```

## Test Case Formats

The service supports two test case formats:

### Legacy Format
```json
{
  "input": "Your question here",
  "expected_output": "Expected answer",
  "metadata": {"key": "value"}
}
```

### LangSmith Format
```json
{
  "inputs": {"question": "Your question here"},
  "outputs": {"answer": "Expected answer"},
  "metadata": {"key": "value"}
}
```

## Usage Examples

### Python SDK Usage

```python
from services.dataset.dataset_service import DatasetService

# Initialize service
dataset_service = DatasetService()

# Create dataset
dataset = await dataset_service.create_dataset(
    name="my_dataset",
    description="Test dataset",
    test_cases=[
        {
            "input": "What is machine learning?",
            "expected_output": "ML is a subset of AI...",
            "metadata": {"topic": "ML"}
        }
    ]
)

# Run dataset
results = await dataset_service.run_dataset(
    dataset_name="my_dataset",
    config={"model": "gpt-4o-mini"}
)
```

### cURL Examples

#### Create Dataset
```bash
curl -X POST http://localhost:5000/api/dataset \\
  -H "Content-Type: application/json" \\
  -d '{
    "dataset_name": "test_dataset",
    "description": "My test dataset",
    "test_cases": [
      {
        "input": "What is AI?",
        "expected_output": "AI is artificial intelligence",
        "metadata": {"category": "basics"}
      }
    ]
  }'
```

#### Get Dataset
```bash
curl -X GET http://localhost:5000/api/dataset/test_dataset
```

#### Run Dataset
```bash
curl -X POST http://localhost:5000/api/dataset/test_dataset/run \\
  -H "Content-Type: application/json" \\
  -d '{
    "config": {
      "model": "gpt-4o-mini",
      "temperature": 0.1
    }
  }'
```

## Response Formats

### Dataset Response
```json
{
  "id": "dataset_uuid",
  "name": "dataset_name",
  "description": "Dataset description",
  "test_cases": [...],
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

### Run Results Response
```json
{
  "dataset_name": "test_dataset",
  "run_id": "run_20250101_120000",
  "config": {...},
  "results": [
    {
      "test_case_index": 0,
      "input": "What is AI?",
      "expected_output": "Expected answer",
      "actual_output": "Actual AI response",
      "success": true,
      "metadata": {...}
    }
  ],
  "summary": {
    "total_tests": 10,
    "successful_tests": 9,
    "failed_tests": 1,
    "success_rate": 0.9
  },
  "timestamp": "2025-01-01T12:00:00Z"
}
```

## Error Handling

The service uses custom exception classes:

- `DatasetNotFoundError`: Raised when a dataset doesn't exist
- `DatasetCreationError`: Raised when dataset creation fails
- `DatasetUpdateError`: Raised when dataset update fails
- `DatasetRunError`: Raised when dataset execution fails

## Integration with Multi-Agent System

The `run_dataset` method is designed to integrate with your multi-agent graph system. Currently, it provides a placeholder implementation. To fully integrate:

1. Import your graph service in the `run_dataset` method
2. Process each test case through your multi-agent pipeline
3. Compare actual outputs with expected outputs
4. Calculate success metrics

## Best Practices

1. **Dataset Organization**: Use descriptive names and categories in metadata
2. **Test Case Design**: Include diverse examples covering edge cases
3. **Batch Operations**: Use bulk operations for better performance
4. **Error Handling**: Always handle potential exceptions in your code
5. **Monitoring**: Use LangSmith's UI to monitor dataset runs and results

## Environment Variables

Required environment variables:
- `LANGSMITH_API_KEY`: Your LangSmith API key
- `LANGSMITH_PROJECT`: Your LangSmith project name (optional)
- `LANGSMITH_ENDPOINT`: LangSmith endpoint (default: https://api.smith.langchain.com)

## Example Script

Run the example script to see the dataset service in action:

```bash
cd examples
python dataset_example.py
```

Make sure to set your LangSmith API key before running the example.
