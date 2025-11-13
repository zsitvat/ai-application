# Graph Service

## Overview

The Graph Service is responsible for multi-agent graph execution using the supervisor pattern. This service handles complex business logic where different AI agents collaborate to solve tasks.

## Main Components

### GraphService

The `GraphService` class provides multi-agent system functionality using the LangGraph framework.

#### Main Features

- Multi-agent orchestration: Coordination of multiple AI agents
- Supervisor pattern: Central control and decision making
- State management: State tracking and persistence
- Streaming support: Real-time responses
- Checkpoint system: Process saving and restoration
- Tool integration: External tool integration
- Personal data filtering: Automatic personal data filtering

## Architecture

### Agent State

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    context: dict
    # Additional state fields...
```

### Workflow Structure

1. START → Supervisor → Agents → Personal Data Filter → END
2. Exception handling: Error handling chain
3. Tool calling: External tool invocation
4. State persistence: State saving to checkpoints
5. Data filtering: Automatic personal data filtering

## Usage

### Initialization

```python
from src.services.graph.graph_service import GraphService
from src.services.data_api.app_settings import AppSettingsService

app_settings_service = AppSettingsService()
graph_service = GraphService(app_settings_service)
```

### Main Methods

#### `execute_graph(user_input, app_id, user_id, context, parameters)`

Graph execution with specified parameters.

Parameters:
- `user_input` (str): User input
- `app_id` (int): Application identifier
- `user_id` (str|None): User identifier
- `context` (dict): Context information
- `parameters` (dict): Runtime parameters

Return Value:
- AI response text

Example:
```python
response = await graph_service.execute_graph(
    user_input="What's the status of the project?",
    app_id=1,
    user_id="user123",
    context={"project_id": "proj_456"},
    parameters={"graph_config": graph_config}
)
```

#### `execute_graph_stream(user_input, app_id, user_id, context, parameters)`

Streaming graph execution for real-time responses.

Parameters: Same as `execute_graph`

Return Value:
- `AsyncGenerator[str, None]`: Streaming response

Example:
```python
async for chunk in graph_service.execute_graph_stream(...):
    print(chunk, end="", flush=True)
```

## Configuration

### Graph Config Structure

```python
class GraphConfig(BaseModel):
    agents: Dict[str, Agent]
    supervisor: dict
    exception_chain: Optional[dict]
    checkpointer_type: CheckpointerType = "memory"
    max_input_length: int = -1
```

### Agent Configuration

```python
class Agent(BaseModel):
    name: str
    enabled: bool = True
    system_prompt: str
    model: Model
    tools: Optional[List[dict]] = None
    temperature: float = 0.7
```

## Checkpoint System

### Supported Types

- Memory: `InMemorySaver` - in-memory storage
- Redis: `RedisSaver` - Redis database storage

### Thread Management

```python
config = {"configurable": {"thread_id": f"{user_id}_{app_id}"}}
```

## Tool Integration

### Tool Loading

```python
def _load_tool_class(self, tool_config: dict):
    module_path = tool_config["module"]
    class_name = tool_config["class"]
    
    module = importlib.import_module(module_path)
    tool_class = getattr(module, class_name)
    return tool_class(**tool_config.get("kwargs", {}))
```

### Supported Tool Types

- Search tools: Vector DB search
- Database tools: SQL queries  
- API tools: External API calls
- Custom tools: Project-specific tools

## Supervisor Logic

### Decision Making

The supervisor decides based on the following logic:

1. User input analysis: Intent recognition
2. Agent selection: Appropriate expert agent
3. Tool usage: Activating necessary tools
4. Response aggregation: Combining multiple responses

### Routing Logic

```python
if "FINAL_ANSWER" in last_message.content:
    return "personal_data_filter"  # Filter before final response
elif needs_tool_call:
    return selected_agent
else:
    return "supervisor"
```

## Personal Data Filter Integration

### Filter Node Implementation

Personal data filtering happens automatically at the end of the workflow, after the applicant_attributes_extractor:

```python
def _personal_data_filter_node(self, state: AgentState) -> AgentState:
    """
    Filter personal data from the final response
    """
    last_message = state["messages"][-1]
    
    # Use PersonalDataFilterService
    filtered_content = self.personal_data_filter.apply_regex_replacements(
        text=last_message.content,
        replacement_patterns=self.personal_data_filter.DEFAULT_PATTERNS
    )
    
    # Update state with filtered content
    filtered_message = HumanMessage(content=filtered_content)
    state["messages"][-1] = filtered_message
    
    return state
```

### Workflow Integration

The filter node is integrated at the following position:

1. Applicant Attributes Extractor → extracts applicant data
2. Personal Data Filter Node → filters personal data
3. END → completes the workflow

### Configuration Options

```python
# Filtering modes configurable in graph config
"personal_data_filter": {
    "enabled": True,
    "use_regex_only": False,      # True = regex only, False = regex + AI
    "replacement_strategy": "mask", # mask, remove, anonymize
    "mask_character": "*"
}
```

## Error Handling

### Exception Chain

Dedicated error handling agent for special error cases:

```python
if "exception_chain" in self.graph_config:
    workflow.add_node("exception_handler", self._exception_node)
```

### Error Reporting

- Detailed logging: Documentation of every step
- Stack trace: Developer information
- User-friendly messages: User-friendly errors

## Streaming Implementation

### Async Generator

```python
async def execute_graph_stream(self, ...):
    async for event in self.workflow.astream(initial_state, config):
        if "supervisor" in event:
            content = event["supervisor"]["messages"][-1].content
            yield content
```

### Real-time Responses

- Chunk-based: Detailed response chunks
- Progress tracking: Progress monitoring
- Error handling: Error handling during streaming

## Performance Optimization

### Parallel Execution

```python
# Agent parallel execution
results = await asyncio.gather(*agent_tasks)
```

### Caching

- State caching: State caching
- Model caching: Model response caching
- Tool result caching: Tool response storage

### Memory Management

- State cleanup: Clearing unnecessary state
- Checkpoint rotation: Deleting old checkpoints
- Message pruning: Message history optimization

## Security Considerations

### Input Validation

- Prompt injection protection: Filtering malicious prompts
- Input length limitation: `max_input_length` validation
- Content filtering: Filtering inappropriate content

### Agent Isolation

- Resource limiting: Resource limitations
- Timeout handling: Protection against overruns
- Error containment: Error isolation

## Logging and Monitoring

### Detailed Logging

- Agent activity: All agent operations
- State changes: State changes
- Performance metrics: Performance indicators
- Error tracking: Error tracking

### Debug Information

- Graph topology: Graph structure
- Message flow: Message flow
- Decision points: Decision points

## Dependencies

- `langgraph`: Graph orchestration
- `langchain_core`: Core LLM functions
- `redis`: State persistence
- `asyncio`: Asynchronous execution
- `importlib`: Dynamic tool loading
