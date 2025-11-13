# Logger Service

## Overview

The Logger Service is responsible for structured application logging. It provides JSON-formatted logging for both console and file output, with rotating files and configurable log levels.

## Main Components

### LoggerService

The `LoggerService` class provides centralized logging functionality for the entire application.

#### Main Features

- Structured logging: JSON-formatted log messages
- **Multi-level output**: Console and file logging
- File rotation: Automatic log file rotation
- Configurable levels: Flexible log level settings
- Singleton pattern: One logger instance per name

### JSONFormatter

Custom JSON formatter for structured logging.

## Usage

### Environment Variables

```bash
LOG_FILE_PATH=/path/to/logs/app.log
LOG_MAX_FILE_SIZE_MB=10
LOG_BACKUP_COUNT=5
```

### Initialization

```python
from src.services.logger.logger_service import LoggerService

logger_service = LoggerService()
logger = logger_service.setup_logger(
    log_level="INFO",
    logger_name="my_application"
)
```

### Basic Usage

```python
logger.info("Application started")
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("An error occurred")
logger.critical("Critical error")
```

## Configuration Options

### Log Levels

- DEBUG: Detailed developer information
- INFO: General informational messages
- WARNING: Warning messages
- ERROR: Error messages
- CRITICAL: Critical errors

### File Configuration

#### File Size Limit

```python
max_bytes = int(os.getenv("LOG_MAX_FILE_SIZE_MB", "10")) * 1024 * 1024
```

#### Number of Backup Files

```python
backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
```

## JSON Output Format

### Standard Fields

```json
{
    "timestamp": "2024-01-15T10:30:45.123Z",
    "level": "INFO",
    "logger": "my_application",
    "message": "Message text",
    "module": "module_name",
    "function": "function_name",
    "line": 42
}
```

### For Errors

```json
{
    "timestamp": "2024-01-15T10:30:45.123Z",
    "level": "ERROR",
    "logger": "my_application", 
    "message": "Error message",
    "exception": {
        "type": "ValueError",
        "message": "Invalid value provided",
        "traceback": "Traceback (most recent call last)..."
    }
}
```

## File Rotation

### RotatingFileHandler

```python
file_handler = RotatingFileHandler(
    log_file_path,
    maxBytes=max_bytes,
    backupCount=backup_count,
    encoding='utf-8'
)
```

### Rotation Logic

- **Size-based**: File reaches maximum size
- Automatic: Start new file and rename old one
- Backup limit: Maximum number of backup files
- **UTF-8 encoding**: Unicode character support

## Singleton Pattern

### Logger Cache

```python
def setup_logger(self, log_level: str = "DEBUG", logger_name: str = "logger"):
    if logger_name in self._loggers:
        return self._loggers[logger_name]
    
    # Create new logger...
    self._loggers[logger_name] = logger
    return logger
```

### Benefits

- Memory efficiency: One instance per logger name
- Consistency: Same settings throughout
- Performance: Faster access

## Error Handling

### File Writing Errors

```python
try:
    self._ensure_log_directory(log_file_path)
    # Create file handler
except Exception as e:
    logger.warning(f"Failed to setup file logging: {e}")
    # Continue with console logging only
```

### Graceful Degradation

- Console logging continues if file logging fails
- Automatic creation of missing directories
- Network drive handling

## Performance Considerations

### Buffering

- Automatic buffer: OS-level optimization
- Flush policy: Immediate writing for critical messages
- Async logging: Non-blocking log writing

### Memory Usage

- Circular buffer: Limited memory usage
- Lazy initialization: Logger creation on-demand
- Cleanup: Automatic resource cleanup

## Security Considerations

### File Permissions

```python
def _ensure_log_directory(self, log_file_path: str):
    log_dir = Path(log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
```

### Sensitive Data

- Data masking: Masking sensitive data
- PII filtering: Personal data filtering
- Secure deletion: Secure log deletion

## Monitoring and Analysis

### Log Aggregation

The JSON format enables:

- ELK Stack: Elasticsearch, Logstash, Kibana
- Splunk: Enterprise log management
- Grafana: Visualization and alerting
- Custom parsers: Custom analysis tools

### Metrics Extraction

```python
# Example: Error rate calculation
error_count = len([log for log in logs if log["level"] == "ERROR"])
total_count = len(logs)
error_rate = error_count / total_count * 100
```

## Integration with Other Services

### FastAPI Integration

```python
import logging
from src.services.logger.logger_service import LoggerService

logger_service = LoggerService()
logger = logger_service.setup_logger("INFO", "api")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response
```

### Database Logging

```python
class DatabaseLogger:
    def __init__(self):
        self.logger = LoggerService().setup_logger("DEBUG", "database")
    
    async def log_query(self, query: str, duration: float):
        self.logger.info("Database query executed", extra={
            "query": query,
            "duration_ms": duration * 1000,
            "operation": "database_query"
        })
```

## Dependencies

- `logging`: Python standard logging
- `logging.handlers`: RotatingFileHandler
- `pathlib`: Path handling
- `sys`: Standard output
- `os`: Környezeti változók
