# Rate Limit Service

## Overview

The Rate Limit Service is responsible for controlling API call rates using the Token Bucket algorithm. This service protects the application from overload and ensures fair usage among different clients.

## Main Components

### TokenBucket

The `TokenBucket` class implements the token bucket algorithm logic for rate limiting.

#### Operating Principle

- **Capacity**: Maximum number of tokens
- **Refill rate**: Token generation per second
- **Token consumption**: Tokens required for request execution
- **Wait time**: Calculated delay when insufficient tokens

### RateLimitMiddleware

FastAPI middleware for limiting incoming HTTP request rates.

#### Main Features

- **IP-based limiting**: Based on client IP addresses
- **Automatic token refill**: Time-based token regeneration
- **Asynchronous waiting**: Non-blocking delay handling
- **Dynamic bucket management**: Client-specific token buckets

## Usage

### Middleware Setup

```python
from fastapi import FastAPI
from src.services.rate_limit.rate_limit import RateLimitMiddleware

app = FastAPI()

# Add rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    capacity=100,      # 100 token capacity
    refill_rate=10.0   # 10 tokens/second refill
)
```

### Configuration Examples

#### Basic API Limiting
```python
# 60 requests/minute
app.add_middleware(RateLimitMiddleware, capacity=60, refill_rate=1.0)
```

#### With Burst Support
```python
# 1000 request burst, then 100 requests/second
app.add_middleware(RateLimitMiddleware, capacity=1000, refill_rate=100.0)
```

#### Strict Limiting
```python
# 10 requests/minute
app.add_middleware(RateLimitMiddleware, capacity=10, refill_rate=0.167)
```

## Token Bucket Algorithm

### Principles

1. **Token pool**: Each client has a token pool
2. **Token consumption**: Each request consumes one token
3. **Automatic refill**: Tokens are regularly replenished
4. **Waiting**: Request waits when insufficient tokens

### Implementation Details

```python
def consume(self, tokens: int = 1) -> float:
    now = time.time()
    elapsed = now - self.last_refill
    
    # Refill tokens based on elapsed time
    self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
    self.last_refill = now
    
    if self.tokens >= tokens:
        self.tokens -= tokens
        return 0.0  # No waiting
    else:
        # Calculate wait time
        required_tokens = tokens - self.tokens
        wait_time = required_tokens / self.refill_rate
        self.tokens = 0
        return wait_time
```

## Middleware Operation

### Request Processing

1. **Client identification**: IP-based bucket retrieval
2. **Token bucket creation**: For new clients
3. **Token consumption**: For request execution
4. **Waiting**: Asynchronous delay when needed
5. **Request forwarding**: After rate limit check

### IP-based Bucket Management

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

## Configuration Parameters

### Capacity

- **Meaning**: Maximum number of tokens in the bucket
- **Effect**: Burst traffic handling
- **Recommendation**: Based on expected peak traffic

### Refill Rate

- **Meaning**: Token generation per second
- **Effect**: Sustained throughput
- **Recommendation**: Based on desired QPS (Queries Per Second)

## Performance Considerations

### Memory Usage

```python
# Bucket cleanup implementation (optional)
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, capacity: int, refill_rate: float, cleanup_interval: int = 3600):
        super().__init__(app)
        self.buckets: dict[str, TokenBucket] = {}
        self.last_cleanup = time.time()
        self.cleanup_interval = cleanup_interval
    
    def _cleanup_old_buckets(self):
        now = time.time()
        if now - self.last_cleanup > self.cleanup_interval:
            # Delete old, inactive buckets
            inactive_ips = [
                ip for ip, bucket in self.buckets.items()
                if now - bucket.last_refill > self.cleanup_interval
            ]
            for ip in inactive_ips:
                del self.buckets[ip]
            self.last_cleanup = now
```

### Scaling Considerations

- **Memory growth**: Bucket count increase with high traffic
- **CPU usage**: Token calculation overhead
- **Network latency**: Effect of wait times

## Error Handling

### Exception Handling

```python
async def dispatch(self, request: Request, call_next):
    try:
        client_ip = request.client.host
        # Rate limiting logic...
        response = await call_next(request)
        return response
    except Exception as e:
        # Allow requests through even with rate limiting errors
        logger.error(f"Rate limiting error: {e}")
        return await call_next(request)
```

### Graceful Degradation

- **On error**: Pass through all requests
- **Partial operation**: Handle token bucket errors
- **Monitoring**: Log errors

## Security Considerations

### DDoS Protection

```python
# Extreme rate limiting against aggressive attacks
class StrictRateLimitMiddleware(RateLimitMiddleware):
    def __init__(self, app):
        super().__init__(app, capacity=10, refill_rate=0.1)  # 6 requests/minute max
```

### IP Spoofing Protection

```python
def get_real_client_ip(self, request: Request) -> str:
    # Handle X-Forwarded-For header for proxy cases
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host
```

## Monitoring and Logging

### Rate Limit Events

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
        
        # Continue middleware...
```

### Metrics Collection

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

## Alternatives and Extensions

### Redis-based Rate Limiting

Scalable solution for multiple server instances:

```python
import redis.asyncio as redis

class RedisRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_url: str, capacity: int, refill_rate: float):
        super().__init__(app)
        self.redis = redis.from_url(redis_url)
        self.capacity = capacity
        self.refill_rate = refill_rate
```

### Key-based Rate Limiting

Rate limiting by API keys instead of IP:

```python
def get_rate_limit_key(self, request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"
    return f"ip:{request.client.host}"
```

## Dependencies

- `asyncio`: Asynchronous operations
- `time`: Time handling
- `fastapi`: Request/Response objects
- `starlette.middleware.base`: Middleware base class
