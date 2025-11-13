# Multi-Agent Graph Execution API

Usage examples for the multi-agent graph execution system with both standard and streaming capabilities.

## Endpoints

### 1. Standard Execution: `/api/graph` (POST)

Executes the multi-agent graph and returns the final result.

Request Body:
```json
{
  "applicationIdentifier": {
    "applicationIdentifier": "1"
  },
  "user_input": "What are the current trends in machine learning?",
  "uuid": "user-123",
  "context": {
    "session_id": "session-456"
  },
  "parameters": {
    "max_tokens": 2000,
    "temperature": 0.7
  }
}
```

Response:
```json
"Based on current research and industry developments, here are the key machine learning trends..."
```

### 2. Streaming Execution: `/api/graph/stream` (POST)

Executes the multi-agent graph with real-time streaming of agent updates and final results.

Request Body: Same as standard execution.

Response: Server-Sent Events (SSE) stream with the following event types:

#### Agent Update Events
```json
data: {
  "event": "agent_update",
  "node": "supervisor", 
  "data": {
    "messages": [
      {
        "type": "humanmessage",
        "content": "What are the current trends in machine learning?"
      }
    ],
    "next_agent": "researcher",
    "user_input": "What are the current trends in machine learning?"
  }
}
```

#### Final Result Event
```json
data: {
  "event": "final_result",
  "data": {
    "response": "Based on current research and industry developments..."
  }
}
```

#### Error Event
```json
data: {
  "event": "error",
  "data": {
    "error": "Connection timeout",
    "fallback_response": "I apologize, but I encountered an error..."
  }
}
```

### 3. Advanced Event Streaming: `/api/graph/stream-events` (POST)

Executes the multi-agent graph with granular event streaming using LangChain's `astream_events`. Provides token-level streaming and detailed component events.

Request Body: Same as standard execution.

Response: Server-Sent Events (SSE) stream with granular event types:

#### Token Events (Real-time token streaming)
```json
data: {
  "event": "token",
  "data": {
    "content": "Based on",
    "model": "ChatOpenAI",
    "run_id": "uuid-1234"
  }
}
```

#### Chain Events
```json
data: {
  "event": "chain_start",
  "data": {
    "name": "RunnableSequence",
    "input": "What are the current trends in ML?"
  }
}

data: {
  "event": "chain_end",
  "data": {
    "name": "RunnableSequence", 
    "output": "Based on current research..."
  }
}
```

#### Component Events
```json
data: {
  "event": "parser_update",
  "data": {"chunk": {"countries": []}},
  "name": "JsonOutputParser"
}
```

### 4. Dynamic Config Execution: `/api/graph/execute-with-config` (POST)

Execute a graph using a dynamically loaded configuration from file or URL.

Request Body:
```json
{
  "user_input": "Analyze this job posting",
  "config_source": "file",
  "config_path": "/path/to/graph_config.json",
  "user_id": "user-123",
  "context": {},
  "parameters": {}
}
```

## Client Examples

### JavaScript/TypeScript (EventSource)

#### Standard Streaming
```javascript
const eventSource = new EventSource('/api/graph/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    applicationIdentifier: { applicationIdentifier: "1" },
    user_input: "What are the current trends in ML?",
    uuid: "user-123"
  })
});

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  
  switch(data.event) {
    case 'agent_update':
      console.log(`Agent ${data.node} update:`, data.data);
      break;
    case 'final_result':
      console.log('Final result:', data.data.response);
      eventSource.close();
      break;
    case 'error':
      console.error('Error:', data.data.error);
      eventSource.close();
      break;
  }
};
```

#### Advanced Event Streaming (Token-level)
```javascript
const eventSource = new EventSource('/api/graph/stream-events', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    applicationIdentifier: { applicationIdentifier: "1" },
    user_input: "What are the current trends in ML?",
    uuid: "user-123"
  })
});

let fullResponse = "";

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  
  switch(data.event) {
    case 'token':
      fullResponse += data.data.content;
      document.getElementById('response').textContent = fullResponse;
      break;
    case 'chain_start':
      console.log(`Chain started: ${data.data.name}`);
      break;
    case 'chain_end':
      console.log(`Chain completed: ${data.data.name}`);
      eventSource.close();
      break;
    case 'error':
      console.error('Error:', data.data.error);
      eventSource.close();
      break;
  }
};
```

### Python (httpx)

```python
import httpx
import json

async def stream_graph_execution():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            'POST',
            'http://localhost:5000/api/graph/stream',
            json={
                "applicationIdentifier": {"applicationIdentifier": "1"},
                "user_input": "What are the current trends in ML?",
                "uuid": "user-123"
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                    
                    if data['event'] == 'agent_update':
                        print(f"Agent {data['node']} update")
                    elif data['event'] == 'final_result':
                        print(f"Final result: {data['data']['response']}")
                        break
                    elif data['event'] == 'error':
                        print(f"Error: {data['data']['error']}")
                        break
```

### cURL

```bash
# Standard execution
curl -X POST "http://localhost:5000/api/graph" \
  -H "Content-Type: application/json" \
  -d '{
    "applicationIdentifier": {"applicationIdentifier": "1"},
    "user_input": "What are the current trends in ML?",
    "uuid": "user-123"
  }'

# Agent-level streaming
curl -X POST "http://localhost:5000/api/graph/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "applicationIdentifier": {"applicationIdentifier": "1"},
    "user_input": "What are the current trends in ML?",
    "uuid": "user-123"
  }' \
  --no-buffer

# Token-level streaming (advanced)
curl -X POST "http://localhost:5000/api/graph/stream-events" \
  -H "Content-Type: application/json" \
  -d '{
    "applicationIdentifier": {"applicationIdentifier": "1"},
    "user_input": "What are the current trends in ML?",
    "uuid": "user-123"
  }' \
  --no-buffer
```

## Configuration

The system supports various graph configurations including:

- OpenAI Models: GPT-4, GPT-3.5-turbo
- Azure OpenAI: Azure-hosted OpenAI models
- Topic Validation: Restrict queries to specific topics
- Personal Data Filtering: Automatic PII detection and filtering
- Custom Agents: Researcher, Writer, Supervisor patterns

See example configurations in the `/examples/` directory:
- `graph_config_example.json` - OpenAI configuration
- `graph_config_azure_openai.json` - Azure OpenAI configuration

## Error Handling

All endpoints include comprehensive error handling:
- Invalid configurations
- Model connection issues
- Rate limiting
- Timeout handling
- Graceful fallback responses

## Rate Limiting

All graph execution endpoints are rate-limited to prevent abuse:
- Default limit: 5 concurrent requests
- Configurable via `RATELIMIT` environment variable
- Applies to `/api/graph`, `/api/graph/stream`, and `/api/graph/stream-events`

## LangChain Streaming Best Practices

The implementation follows LangChain's recommended streaming patterns:

1. `astream()` for Graph Streaming: Uses LangGraph's `astream()` for step-by-step execution streaming
2. AsyncGenerator Pattern: Proper async generator implementation for efficient streaming
3. Event-Based Architecture: Structured events for different stages of execution
4. Server-Sent Events: Standard SSE format for web compatibility
5. Error Handling: Graceful error handling with fallback responses

For more advanced streaming needs (token-level events), the system can be extended with `astream_events()` for granular control over streaming events.
