import asyncio
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> float:
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return 0.0  # No wait needed
        else:
            required_tokens = tokens - self.tokens
            wait_time = required_tokens / self.refill_rate
            self.tokens = 0
            return wait_time


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, capacity: int, refill_rate: float):
        super().__init__(app)
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets: dict[str, TokenBucket] = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client is not None else "unknown"
        bucket = self.buckets.get(client_ip)

        if not bucket:
            bucket = TokenBucket(self.capacity, self.refill_rate)
            self.buckets[client_ip] = bucket

        wait_time = bucket.consume()
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        response = await call_next(request)
        return response
