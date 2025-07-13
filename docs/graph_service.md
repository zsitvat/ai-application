# Graph Service

## Áttekintés

A Graph Service felelős a többügynökös (multi-agent) gráf végrehajtásáért supervisor mintázat használatával. Ez a service kezeli az összetett üzleti logikát, ahol különböző AI ügynökök együttműködnek a feladatok megoldásában.

## Főbb komponensek

### GraphService

A `GraphService` osztály a LangGraph keretrendszert használva biztosítja a multi-agent rendszer funkcionalitását.

#### Főbb funkciók

- **Multi-agent orchestration**: Több AI ügynök koordinálása
- **Supervisor pattern**: Központi irányítás és döntéshozatal
- **State management**: Állapot követése és mentése
- **Streaming támogatás**: Valós idejű válaszok
- **Checkpoint rendszer**: Folyamat mentés és helyreállítás
- **Tool integration**: Külső eszközök beépítése

## Architektúra

### Agent State

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str
    context: dict
    # További állapot mezők...
```

### Workflow struktúra

1. **START** → **Supervisor** → **Agents** → **END**
2. **Exception handling**: Hibakezelő chain
3. **Tool calling**: Külső eszközök hívása
4. **State persistence**: Állapot mentés checkpointokba

## Használat

### Inicializálás

```python
from src.services.graph.graph_service import GraphService
from src.services.data_api.app_settings import AppSettingsService

app_settings_service = AppSettingsService()
graph_service = GraphService(app_settings_service)
```

### Főbb metódusok

#### `execute_graph(user_input, app_id, user_id, context, parameters)`

Gráf végrehajtása megadott paraméterekkel.

**Paraméterek:**
- `user_input` (str): Felhasználói bemenet
- `app_id` (int): Alkalmazás azonosító
- `user_id` (str|None): Felhasználó azonosító
- `context` (dict): Kontextus információk
- `parameters` (dict): Futtatási paraméterek

**Visszatérési érték:**
- AI válasz szöveg

**Példa:**
```python
response = await graph_service.execute_graph(
    user_input="Mi a helyzet a projekttel?",
    app_id=1,
    user_id="user123",
    context={"project_id": "proj_456"},
    parameters={"graph_config": graph_config}
)
```

#### `execute_graph_stream(user_input, app_id, user_id, context, parameters)`

Streaming gráf végrehajtás valós idejű válaszokhoz.

**Paraméterek:** Ugyanazok mint az `execute_graph`-nál

**Visszatérési érték:** 
- `AsyncGenerator[str, None]`: Streaming válasz

**Példa:**
```python
async for chunk in graph_service.execute_graph_stream(...):
    print(chunk, end="", flush=True)
```

## Konfiguráció

### Graph Config struktúra

```python
class GraphConfig(BaseModel):
    agents: Dict[str, Agent]
    supervisor: dict
    exception_chain: Optional[dict]
    checkpointer_type: CheckpointerType = "memory"
    max_input_length: int = -1
```

### Agent konfiguráció

```python
class Agent(BaseModel):
    name: str
    enabled: bool = True
    system_prompt: str
    model: Model
    tools: Optional[List[dict]] = None
    temperature: float = 0.7
```

## Checkpoint rendszer

### Támogatott típusok

- **Memory**: `InMemorySaver` - memóriában tárolás
- **Redis**: `RedisSaver` - Redis adatbázisban tárolás

### Thread kezelés

```python
config = {"configurable": {"thread_id": f"{user_id}_{app_id}"}}
```

## Tool integráció

### Tool loading

```python
def _load_tool_class(self, tool_config: dict):
    module_path = tool_config["module"]
    class_name = tool_config["class"]
    
    module = importlib.import_module(module_path)
    tool_class = getattr(module, class_name)
    return tool_class(**tool_config.get("kwargs", {}))
```

### Támogatott tool típusok

- **Keresési eszközök**: Vector DB keresés
- **Adatbázis eszközök**: SQL lekérdezések  
- **API eszközök**: Külső API hívások
- **Egyedi eszközök**: Projekt-specifikus tools

## Supervisor logika

### Döntéshozatal

A supervisor a következő logika alapján dönt:

1. **User input elemzés**: Szándék felismerés
2. **Agent kiválasztás**: Megfelelő szakértő agent
3. **Tool használat**: Szükséges eszközök aktiválása
4. **Válasz aggregálás**: Többes válaszok összegzése

### Routing logika

```python
if "FINAL_ANSWER" in last_message.content:
    return "end"
elif needs_tool_call:
    return selected_agent
else:
    return "supervisor"
```

## Hibakezelés

### Exception Chain

Dedikált hibakezelő ügynök speciális hibák kezelésére:

```python
if "exception_chain" in self.graph_config:
    workflow.add_node("exception_handler", self._exception_node)
```

### Hibajelentés

- **Részletes naplózás**: Minden lépés dokumentálása
- **Stack trace**: Fejlesztői információk
- **User-friendly üzenetek**: Felhasználóbarát hibák

## Streaming implementáció

### Async Generator

```python
async def execute_graph_stream(self, ...):
    async for event in self.workflow.astream(initial_state, config):
        if "supervisor" in event:
            content = event["supervisor"]["messages"][-1].content
            yield content
```

### Real-time válaszok

- **Chunk-based**: Részletes válasz darabok
- **Progress tracking**: Előrehaladás követése
- **Error handling**: Hibák streaming közben

## Teljesítmény optimalizáció

### Párhuzamos végrehajtás

```python
# Agent párhuzamos futtatás
results = await asyncio.gather(*agent_tasks)
```

### Caching

- **State caching**: Állapot gyorsítótárazás
- **Model caching**: Modell válaszok cache-elése
- **Tool result caching**: Eszköz válaszok tárolása

### Memory management

- **State cleanup**: Felesleges állapot törlése
- **Checkpoint rotation**: Régi checkpointok törlése
- **Message pruning**: Üzenet történet optimalizálás

## Biztonsági szempontok

### Input validáció

- **Prompt injection védelem**: Káros promptok szűrése
- **Input hossz korlátozás**: `max_input_length` ellenőrzés
- **Content filtering**: Nem megfelelő tartalom szűrése

### Agent izolálás

- **Resource limiting**: Erőforrás korlátozások
- **Timeout handling**: Túlfutás elleni védelem
- **Error containment**: Hibák izolálása

## Naplózás és monitorozás

### Részletes naplózás

- **Agent activity**: Minden agent művelet
- **State changes**: Állapot változások
- **Performance metrics**: Teljesítmény mutatók
- **Error tracking**: Hibák nyomon követése

### Debug információk

- **Graph topology**: Gráf struktúra
- **Message flow**: Üzenet áramlás
- **Decision points**: Döntési pontok

## Függőségek

- `langgraph`: Graph orchestration
- `langchain_core`: Core LLM funkciók
- `redis`: State persistence
- `asyncio`: Aszinkron végrehajtás
- `importlib`: Dinamikus tool loading
