# System Service

## Main Components

### SystemService

The `SystemService` class provides comprehensive system monitoring and diagnostic functionality.

#### Main Functions

- Health check: Application and system status verification
- System metrics: CPU, memory, disk usage monitoring
- Environment variables: Configuration validation
- External services: Dependencies status verification
- Log management: Log files listing and content retrieval

## Usage

### Initialization

```python
from src.services.system.system_service import SystemService

system_service = SystemService()
```

### Main Methods

#### `health_check()`

Performs comprehensive application health check.

Return Value:
- Dictionary with detailed health information

Example:
```python
health_status = await system_service.health_check()
print(health_status)
```

Response Structure:
```python
{
    "status": "healthy",  # "healthy", "degraded", "unhealthy"
    "timestamp": "2024-01-15T10:30:45.123Z",
    "response_time_ms": 156.7,
    "checks": {
        "system": {...},
        "environment": {...},
        "services": {...}
    },
    "version": "1.0.0"
}
```

#### `get_log_files()`

Lists available log files.

Return Value:
- List of log file names

#### `get_log_content(filename, lines)`

Retrieves log file content.

Parameters:
- `filename` (str): Log file name
- `lines` (int): Number of lines to return (last N lines)

Return Value:
- Log content in string format

## Health Check Components

### System Metrics

```python
async def _check_system_metrics(self) -> dict:
    return {
        "status": "healthy",
        "cpu_usage": psutil.cpu_percent(interval=1),
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent
        },
        "disk": {
            "total": psutil.disk_usage('/').total,
            "free": psutil.disk_usage('/').free,
            "percent": psutil.disk_usage('/').percent
        }
    }
```

### Environment Variables Check

```python
async def _check_environment_variables(self) -> dict:
    required_vars = [
        "DATA_API_BASE_URL",
        "REDIS_HOST",
        "REDIS_PORT",
        # Additional required variables...
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    return {
        "status": "healthy" if not missing_vars else "degraded",
        "missing_variables": missing_vars,
        "total_required": len(required_vars),
        "configured": len(required_vars) - len(missing_vars)
    }
```

### External Services

```python
async def _check_external_services(self) -> dict:
    services = {}
    
    # Redis connection check
    services["redis"] = await self._check_redis_connection()
    
    # Data API check
    services["data_api"] = await self._check_data_api_connection()
    
    # Additional services...
    
    healthy_services = sum(1 for s in services.values() if s["status"] == "healthy")
    
    return {
        "status": "healthy" if healthy_services == len(services) else "degraded",
        "services": services,
        "healthy_count": healthy_services,
        "total_count": len(services)
    }
```

## Log Management

### Log Files Discovery

```python
async def get_log_files(self) -> list[str]:
    log_directory = Path(os.getenv("LOG_FILE_PATH", "logs")).parent
    
    if not log_directory.exists():
        return []
    
    log_files = []
    for file_path in log_directory.glob("*.log*"):
        if file_path.is_file():
            log_files.append(file_path.name)
    
    return sorted(log_files, reverse=True)  # Latest first
```

### Log Content Reading

```python
async def get_log_content(self, filename: str, lines: int = 100) -> str:
    log_directory = Path(os.getenv("LOG_FILE_PATH", "logs")).parent
    file_path = log_directory / filename
    
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Log file not found: {filename}")
    
    # Read last N lines
    with open(file_path, 'r', encoding='utf-8') as file:
        return ''.join(file.readlines()[-lines:])
```

## System Metrics

### CPU Usage

```python
cpu_percent = psutil.cpu_percent(interval=1)  # 1 second measurement
cpu_count = psutil.cpu_count()
load_average = os.getloadavg()  # Unix/Linux only
```

### Memory Usage

```python
memory = psutil.virtual_memory()
memory_info = {
    "total": memory.total,
    "available": memory.available,
    "percent": memory.percent,
    "used": memory.used,
    "free": memory.free
}
```

### Disk Usage

```python
disk = psutil.disk_usage('/')
disk_info = {
    "total": disk.total,
    "used": disk.used,
    "free": disk.free,
    "percent": (disk.used / disk.total) * 100
}
```

## External Service Checks

### Redis Connection

```python
async def _check_redis_connection(self) -> dict:
    try:
        redis_url = f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}"
        redis_client = redis.from_url(redis_url)
        
        await redis_client.ping()
        info = await redis_client.info()
        
        return {
            "status": "healthy",
            "response_time_ms": response_time,
            "version": info.get("redis_version"),
            "connected_clients": info.get("connected_clients"),
            "used_memory": info.get("used_memory_human")
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

### Data API Connection

```python
async def _check_data_api_connection(self) -> dict:
    try:
        base_url = os.getenv("DATA_API_BASE_URL")
        health_endpoint = f"{base_url}/health"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(health_endpoint, timeout=5.0)
            
        return {
            "status": "healthy" if response.status_code == 200 else "degraded",
            "status_code": response.status_code,
            "response_time_ms": response_time
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

## Performance Monitoring

### Response Time Measurement

```python
async def health_check(self) -> dict:
    start_time = time.time()
    
    # Health check operations...
    
    response_time = round((time.time() - start_time) * 1000, 2)
    
    return {
        "response_time_ms": response_time,
        # Additional results...
    }
```

### Metrics Collection

```python
class MetricsCollector:
    def __init__(self):
        self.health_check_count = 0
        self.avg_response_time = 0
        self.last_check_timestamp = None
    
    def record_health_check(self, response_time: float):
        self.health_check_count += 1
        self.avg_response_time = (
            (self.avg_response_time * (self.health_check_count - 1) + response_time)
            / self.health_check_count
        )
        self.last_check_timestamp = datetime.now()
```

## Error Handling

### Graceful degradation

```python
async def health_check(self) -> dict:
    checks = {}
    overall_status = "healthy"
    
    try:
        checks["system"] = await self._check_system_metrics()
    except Exception as e:
        checks["system"] = {"status": "error", "error": str(e)}
        overall_status = "degraded"
    
    # Additional checks similarly...
    
    return {
        "status": overall_status,
        "checks": checks
    }
```

### Timeout Handling

```python
async def _check_with_timeout(self, check_func, timeout: float = 5.0):
    try:
        return await asyncio.wait_for(check_func(), timeout=timeout)
    except asyncio.TimeoutError:
        return {"status": "timeout", "error": "Check timed out"}
```

## Security Considerations

### Log File Access

```python
def _validate_log_filename(self, filename: str) -> bool:
    # Protection against path traversal attacks
    if ".." in filename or "/" in filename or "\\" in filename:
        return False
    
    # Only .log extension files
    if not filename.endswith(('.log', '.log.1', '.log.2')):
        return False
    
    return True
```

### Sensitive Information Filtering

```python
def _sanitize_environment_info(self, env_info: dict) -> dict:
    sensitive_keys = ["password", "secret", "key", "token"]
    
    for key in env_info:
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            env_info[key] = "***REDACTED***"
    
    return env_info
```

## Integration with Other Services

### FastAPI health endpoint

```python
from fastapi import APIRouter
from src.services.system.system_service import SystemService

router = APIRouter()
system_service = SystemService()

@router.get("/health")
async def health_check():
    return await system_service.health_check()

@router.get("/logs")
async def get_log_files():
    return await system_service.get_log_files()

@router.get("/logs/{filename}")
async def get_log_content(filename: str, lines: int = 100):
    return await system_service.get_log_content(filename, lines)
```

## Dependencies

- `psutil`: System metrics (CPU, memory, disk)
- `pathlib`: File path handling
- `asyncio`: Asynchronous operations
- `time`: Time measurement
- `datetime`: Timestamps
- `os`: Environment variables and system info
