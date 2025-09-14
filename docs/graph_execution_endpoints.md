# Graph Execution Endpoints

This document describes the two graph execution endpoints available in the recruiter AI application.

## Endpoints Overview

### 1. Standard Graph Execution
**Endpoint:** `POST /api/graph`  
**Response Type:** JSON String  
**Use Case:** When you need the complete result after all agents have finished processing.

### 2. Streaming Graph Execution  
**Endpoint:** `POST /api/graph/stream`  
**Response Type:** Server-Sent Events (SSE)  
**Use Case:** When you want real-time updates as each agent processes the request.

## Request Schema

Both endpoints use the same request schema (`RestOperationPostSchema`):

```json
{
  "applicationIdentifier": {
    "applicationIdentifier": "1"
  },
  "uuid": "user-session-id",
  "user_input": "Your question or input text",
  "platform": "WEBCHAT",
  "context": {},
  "parameters": {}
}
```

### Request Fields

- **applicationIdentifier**: Object containing the app ID as a string
- **uuid**: Unique identifier for the user session
- **user_input**: The user's question or input text
- **platform**: Platform type (defaults to "WEBCHAT")
- **context**: Additional context dictionary (optional)
- **parameters**: Runtime parameters including graph configurations (optional)

## Response Formats

### Standard Endpoint Response

Returns a simple JSON string with the final response:

```json
"The final response from all agents after processing is complete."
```

### Streaming Endpoint Response

Returns Server-Sent Events with the following event types:

#### Agent Update Event
```
data: {
  "event": "agent_update",
  "node": "agent_name",
  "data": {
    "messages": [
      {
        "type": "humanmessage",
        "content": "User input or agent response"
      }
    ],
    "next_agent": "next_agent_name",
    "user_input": "Original user input"
  }
}
```

#### Final Result Event
```
data: {
  "event": "final_result",
  "data": {
    "response": "The complete final response"
  }
}
```

#### Error Event
```
data: {
  "event": "error",
  "data": {
    "error": "Error description",
    "fallback_response": "Fallback response if available"
  }
}
```

## Usage Examples

### JavaScript/Browser

```javascript
// Standard execution
async function executeGraph(requestData) {
    const response = await fetch('/api/graph', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    });
    return await response.json();
}

// Streaming execution
function executeGraphStream(requestData) {
    const eventSource = new EventSource('/api/graph/stream');
    
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        switch(data.event) {
            case 'agent_update':
                console.log(`Agent ${data.node} is processing...`);
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
}
```

### Python

```python
import requests
import json

# Standard execution
def execute_graph(request_data):
    response = requests.post('/api/graph', json=request_data)
    return response.json()

# Streaming execution
def execute_graph_stream(request_data):
    with requests.post('/api/graph/stream', json=request_data, stream=True) as response:
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                event_data = json.loads(line[6:])
                yield event_data
```

### cURL

```bash
# Standard execution
curl -X POST "http://localhost:8000/api/graph" \
  -H "Content-Type: application/json" \
  -d '{
    "applicationIdentifier": {"applicationIdentifier": "1"},
    "uuid": "test-user",
    "user_input": "What is AI?",
    "platform": "WEBCHAT"
  }'

# Streaming execution
curl -X POST "http://localhost:8000/api/graph/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "applicationIdentifier": {"applicationIdentifier": "1"},
    "uuid": "test-user",
    "user_input": "What is AI?",
    "platform": "WEBCHAT"
  }'
```

## Error Handling

Both endpoints return HTTP 500 with error details in case of failures:

```json
{
  "detail": "Error executing multi-agent graph: Detailed error message"
}
```

For the streaming endpoint, errors are also sent as SSE events before the connection closes.

## Configuration

Graph configurations can be provided in the request parameters or loaded from app settings:

```json
{
  "parameters": {
    "graph_config": {
      "agents": {
        "agent_name": {
          "enabled": true,
          "chain": {
            "model": {
              "provider": "openai",
              "name": "gpt-4o-mini",
              "deployment": "gpt-4o-mini"
            },
            "prompt": "Your agent's system prompt"
          }
        }
      },
      "allow_supervisor_finish": true
    }
  }
}
```

## Performance Considerations

- **Standard endpoint**: Use when you only need the final result and don't require real-time updates
- **Streaming endpoint**: Use when you want to show progress to users or need intermediate results
- The streaming endpoint may use more resources due to maintaining open connections
- Consider implementing proper timeout and reconnection logic for streaming clients
