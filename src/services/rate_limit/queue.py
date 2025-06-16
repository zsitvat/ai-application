import asyncio
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class SequentialProcessingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next):
        async with self.lock:
            response = await call_next(request)
            return response
