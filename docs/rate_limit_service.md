# Rate Limit Service

## Áttekintés

A Rate Limit Service felelős az API hívások sebességének korlátozásáért Token Bucket algoritmus használatával. Ez a service védi az alkalmazást a túlterheléstől és biztosítja a fair használatot a különböző kliensek között.

## Főbb komponensek

### TokenBucket

A `TokenBucket` osztály implementálja a token bucket algoritmus logikáját a rate limiting számára.

#### Működési elv

- **Kapacitás**: Maximális token szám
- **Feltöltési ráta**: Token-ek másodpercenkénti generálása
- **Token fogyasztás**: Kérés végrehajtáshoz szükséges token-ek
- **Várakozási idő**: Nem elég token esetén számított delay

### RateLimitMiddleware

FastAPI middleware a bejövő HTTP kérések sebességének korlátozására.

#### Főbb funkciók

- **IP-alapú korlátozás**: Kliens IP címek szerint
- **Automatikus token feltöltés**: Időalapú token regenerálás
- **Aszinkron várakozás**: Non-blocking delay kezelés
- **Dinamikus bucket kezelés**: Kliens-specifikus token buckets

## Használat

### Middleware beállítás

```python
from fastapi import FastAPI
from src.services.rate_limit.rate_limit import RateLimitMiddleware

app = FastAPI()

# Rate limiting middleware hozzáadása
app.add_middleware(
    RateLimitMiddleware,
    capacity=100,      # 100 token kapacitás
    refill_rate=10.0   # 10 token/másodperc feltöltés
)
```

### Konfiguráció példák

#### Alapvető API korlátozás
```python
# 60 kérés/perc
app.add_middleware(RateLimitMiddleware, capacity=60, refill_rate=1.0)
```

#### Burst támogatással
```python
# 1000 kérés burst, majd 100 kérés/másodperc
app.add_middleware(RateLimitMiddleware, capacity=1000, refill_rate=100.0)
```

#### Szigorú korlátozás
```python
# 10 kérés/perc
app.add_middleware(RateLimitMiddleware, capacity=10, refill_rate=0.167)
```

## Token Bucket algoritmus

### Alapelvek

1. **Token készlet**: Minden kliens rendelkezik token készlettel
2. **Token fogyasztás**: Minden kérés elvesz egy token-t
3. **Automatikus feltöltés**: Token-ek rendszeresen pótlódnak
4. **Várakozás**: Nincs elég token esetén a kérés várakozik

### Implementáció részletek

```python
def consume(self, tokens: int = 1) -> float:
    now = time.time()
    elapsed = now - self.last_refill
    
    # Token-ek pótlása az eltelt idő alapján
    self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
    self.last_refill = now
    
    if self.tokens >= tokens:
        self.tokens -= tokens
        return 0.0  # Nincs várakozás
    else:
        # Várakozási idő számítása
        required_tokens = tokens - self.tokens
        wait_time = required_tokens / self.refill_rate
        self.tokens = 0
        return wait_time
```

## Middleware működés

### Request feldolgozás

1. **Kliens azonosítás**: IP cím alapú bucket lekérés
2. **Token bucket létrehozás**: Új kliens esetén
3. **Token fogyasztás**: Kérés végrehajtásához
4. **Várakozás**: Szükség esetén aszinkron delay
5. **Request továbbítás**: Rate limit után

### IP-alapú bucket kezelés

```python
async def dispatch(self, request: Request, call_next):
    client_ip = request.client.host
    bucket = self.buckets.get(client_ip)
    
    if not bucket:
        bucket = TokenBucket(self.capacity, self.refill_rate)
        self.buckets[client_ip] = bucket
    
    wait_time = bucket.consume()
    if wait_time > 0:
        await asyncio.sleep(wait_time)
    
    response = await call_next(request)
    return response
```

## Konfigurációs paraméterek

### Capacity (Kapacitás)

- **Jelentés**: Maximális token szám a bucket-ben
- **Hatás**: Burst forgalom kezelése
- **Ajánlás**: Várható peak forgalom alapján

### Refill Rate (Feltöltési ráta)

- **Jelentés**: Token-ek másodpercenkénti generálása
- **Hatás**: Tartós áteresztőképesség
- **Ajánlás**: Kívánt QPS (Queries Per Second) alapján

## Teljesítmény szempontok

### Memória használat

```python
# Bucket cleanup implementáció (opcionális)
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, capacity: int, refill_rate: float, cleanup_interval: int = 3600):
        super().__init__(app)
        self.buckets: dict[str, TokenBucket] = {}
        self.last_cleanup = time.time()
        self.cleanup_interval = cleanup_interval
    
    def _cleanup_old_buckets(self):
        now = time.time()
        if now - self.last_cleanup > self.cleanup_interval:
            # Régi, inaktív bucket-ek törlése
            inactive_ips = [
                ip for ip, bucket in self.buckets.items()
                if now - bucket.last_refill > self.cleanup_interval
            ]
            for ip in inactive_ips:
                del self.buckets[ip]
            self.last_cleanup = now
```

### Scaling megfontolások

- **Memory growth**: Bucket szám növekedés nagy forgalomnál
- **CPU usage**: Token számítások overhead
- **Network latency**: Várakozási idők hatása

## Hibakezelés

### Exception handling

```python
async def dispatch(self, request: Request, call_next):
    try:
        client_ip = request.client.host
        # Rate limiting logika...
        response = await call_next(request)
        return response
    except Exception as e:
        # Hibás rate limiting esetén is engedjük át a kérést
        logger.error(f"Rate limiting error: {e}")
        return await call_next(request)
```

### Graceful degradation

- **Hiba esetén**: Átenged minden kérést
- **Részleges működés**: Token bucket hibák esetén
- **Monitoring**: Hibák naplózása

## Biztonsági szempontok

### DDoS védelem

```python
# Extrém rate limiting agresszív támadások ellen
class StrictRateLimitMiddleware(RateLimitMiddleware):
    def __init__(self, app):
        super().__init__(app, capacity=10, refill_rate=0.1)  # 6 kérés/perc max
```

### IP spoofing védelem

```python
def get_real_client_ip(self, request: Request) -> str:
    # X-Forwarded-For header kezelése proxy esetén
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host
```

## Monitorozás és naplózás

### Rate limit események

```python
import logging

logger = logging.getLogger(__name__)

class MonitoredRateLimitMiddleware(RateLimitMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        bucket = self.buckets.get(client_ip)
        
        wait_time = bucket.consume() if bucket else 0
        
        if wait_time > 0:
            logger.warning(f"Rate limit applied for {client_ip}, wait: {wait_time}s")
        
        # Middleware folytatása...
```

### Metrikák gyűjtése

```python
class MetricsRateLimitMiddleware(RateLimitMiddleware):
    def __init__(self, app, capacity: int, refill_rate: float):
        super().__init__(app, capacity, refill_rate)
        self.total_requests = 0
        self.rate_limited_requests = 0
    
    async def dispatch(self, request: Request, call_next):
        self.total_requests += 1
        
        wait_time = bucket.consume()
        if wait_time > 0:
            self.rate_limited_requests += 1
        
        # Rate limiting logic...
```

## Alternatívák és kiterjesztések

### Redis-alapú rate limiting

Skálázható megoldás több server instance esetén:

```python
import redis.asyncio as redis

class RedisRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_url: str, capacity: int, refill_rate: float):
        super().__init__(app)
        self.redis = redis.from_url(redis_url)
        self.capacity = capacity
        self.refill_rate = refill_rate
```

### Kulcs-alapú rate limiting

API kulcsok szerint korlátozás IP helyett:

```python
def get_rate_limit_key(self, request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"
    return f"ip:{request.client.host}"
```

## Függőségek

- `asyncio`: Aszinkron műveletek
- `time`: Időkezelés
- `fastapi`: Request/Response objektumok
- `starlette.middleware.base`: Middleware alaposztály
