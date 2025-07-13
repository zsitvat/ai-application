# Logger Service

## Áttekintés

A Logger Service felelős az alkalmazás strukturált naplózásáért. JSON formátumú naplózást biztosít mind konzol, mind fájl kimenethez, forgatható fájlokkal és konfigurálható naplózási szintekkel.

## Főbb komponensek

### LoggerService

A `LoggerService` osztály centralizált naplózási funkcionalitást biztosít az egész alkalmazás számára.

#### Főbb funkciók

- **Strukturált naplózás**: JSON formátumú log üzenetek
- **Többszintű kimenet**: Konzol és fájl logging
- **Fájl rotáció**: Automatikus log fájl forgatás
- **Konfigurálható szintek**: Rugalmas log level beállítások
- **Singleton pattern**: Egy logger instance per név

### JSONFormatter

Egyedi JSON formatter a strukturált naplózáshoz.

## Használat

### Környezeti változók

```bash
LOG_FILE_PATH=/path/to/logs/app.log
LOG_MAX_FILE_SIZE_MB=10
LOG_BACKUP_COUNT=5
```

### Inicializálás

```python
from src.services.logger.logger_service import LoggerService

logger_service = LoggerService()
logger = logger_service.setup_logger(
    log_level="INFO",
    logger_name="my_application"
)
```

### Alapvető használat

```python
logger.info("Alkalmazás elindult")
logger.debug("Debug információ")
logger.warning("Figyelmeztetés")
logger.error("Hiba történt")
logger.critical("Kritikus hiba")
```

## Konfigurációs opciók

### Log szintek

- **DEBUG**: Részletes fejlesztői információk
- **INFO**: Általános információs üzenetek
- **WARNING**: Figyelmeztetések
- **ERROR**: Hibák
- **CRITICAL**: Kritikus hibák

### Fájl konfiguráció

#### Fájl méret korlát

```python
max_bytes = int(os.getenv("LOG_MAX_FILE_SIZE_MB", "10")) * 1024 * 1024
```

#### Backup fájlok száma

```python
backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
```

## JSON kimenet formátum

### Standard mezők

```json
{
    "timestamp": "2024-01-15T10:30:45.123Z",
    "level": "INFO",
    "logger": "my_application",
    "message": "Üzenet szövege",
    "module": "module_name",
    "function": "function_name",
    "line": 42
}
```

### Hibák esetén

```json
{
    "timestamp": "2024-01-15T10:30:45.123Z",
    "level": "ERROR",
    "logger": "my_application", 
    "message": "Hiba üzenet",
    "exception": {
        "type": "ValueError",
        "message": "Invalid value provided",
        "traceback": "Traceback (most recent call last)..."
    }
}
```

## Fájl rotáció

### RotatingFileHandler

```python
file_handler = RotatingFileHandler(
    log_file_path,
    maxBytes=max_bytes,
    backupCount=backup_count,
    encoding='utf-8'
)
```

### Rotációs logika

- **Méret alapú**: Fájl eléri a max méretet
- **Automatikus**: Új fájl indítása és régi átnevezése
- **Backup limit**: Maximális backup fájlok száma
- **UTF-8 encoding**: Unicode karakterek támogatása

## Singleton pattern

### Logger cache

```python
def setup_logger(self, log_level: str = "DEBUG", logger_name: str = "logger"):
    if logger_name in self._loggers:
        return self._loggers[logger_name]
    
    # Új logger létrehozása...
    self._loggers[logger_name] = logger
    return logger
```

### Előnyök

- **Memória hatékonyság**: Egy instance per logger név
- **Konzisztencia**: Ugyanazok a beállítások
- **Performance**: Gyorsabb hozzáférés

## Hibakezelés

### Fájl írási hibák

```python
try:
    self._ensure_log_directory(log_file_path)
    # Fájl handler létrehozása
except Exception as e:
    logger.warning(f"Failed to setup file logging: {e}")
    # Folytatás csak konzol logginggal
```

### Graceful degradation

- Fájl logging hiba esetén konzol logging folytatódik
- Hiányzó könyvtárak automatikus létrehozása
- Hálózati meghajtók kezelése

## Teljesítmény szempontok

### Buffering

- **Automatikus buffer**: OS szintű optimalizáció
- **Flush policy**: Kritikus üzenetek azonnali írása
- **Async logging**: Non-blocking log írás

### Memory usage

- **Circular buffer**: Korlátozott memória használat
- **Lazy initialization**: Logger on-demand létrehozás
- **Cleanup**: Automatikus resource felszabadítás

## Biztonsági szempontok

### Fájl jogosultságok

```python
def _ensure_log_directory(self, log_file_path: str):
    log_dir = Path(log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
```

### Sensitive data

- **Data masking**: Érzékeny adatok maszkolása
- **PII filtering**: Személyes adatok szűrése
- **Secure deletion**: Biztonságos log törlés

## Monitorozás és elemzés

### Log aggregáció

A JSON formátum lehetővé teszi:

- **ELK Stack**: Elasticsearch, Logstash, Kibana
- **Splunk**: Enterprise log management
- **Grafana**: Vizualizáció és alerting
- **Custom parsers**: Egyedi elemző eszközök

### Metrikák kinyerése

```python
# Példa: Error rate számítás
error_count = len([log for log in logs if log["level"] == "ERROR"])
total_count = len(logs)
error_rate = error_count / total_count * 100
```

## Integráció más szolgáltatásokkal

### FastAPI integration

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

### Database logging

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

## Függőségek

- `logging`: Python standard logging
- `logging.handlers`: RotatingFileHandler
- `pathlib`: Path kezelés
- `sys`: Standard output
- `os`: Környezeti változók
