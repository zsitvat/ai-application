# Chat History Service

## Overview

The Chat History Service is responsible for managing user conversation histories. It uses Redis database for storing and retrieving chat histories.

## Main Components

### RedisChatHistoryService

The `RedisChatHistoryService` class provides chat history management in Redis database.

#### Main Features

- **Chat history retrieval**: Retrieving histories based on user
- **Conversation data storage**: Saving new messages to Redis
- **Session management**: Managing different conversation sessions
- **Metadata storage**: Title, timestamp and other information handling

## Usage

### Environment Variables

```bash
RAG_REDIS_USER=your_redis_username
RAG_REDIS_PASSWORD=your_redis_password
RAG_REDIS_HOST=localhost
RAG_REDIS_PORT=6379
```

### Initialization

```python
from src.services.chat_history.redis_chat_history import RedisChatHistoryService

chat_service = RedisChatHistoryService()
```

### Main Methods

#### `get_chat_histories(chat_memory_db, chat_memory_index_name, user_id)`

Retrieves all chat histories for a user.

**Parameters:**
- `chat_memory_db` (str): Redis database name
- `chat_memory_index_name` (str): Index name in Redis
- `user_id` (str): User identifier

**Return Value:**
- List of chat histories containing session ID, title and timestamp

## Redis Data Structure

The service uses the following key format:
```
{chat_memory_index_name}:{user_id}:{session_id}
```

Each key contains a list with JSON formatted messages.

## Error Handling

- **Redis connection errors**: Automatic reconnection and error reporting
- **Data format errors**: JSON parsing error handling
- **Missing data**: Graceful handling of missing or corrupted data

## Logging

The service provides detailed logging:
- Debug level information about Redis connections
- User operation tracking
- Error reports with detailed information

## Performance

- **Batch operations**: Efficient handling of large amounts of data
- **Cursor-based pagination**: Memory-efficient data search
- **Asynchronous operations**: Non-blocking I/O operations

## Security Considerations

- Password-protected Redis connections
- User data isolation
- Secure key generation
