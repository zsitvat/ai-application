import asyncio
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class SemaphoreMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, semaphore_limit: int = 10, paths_to_limit: list = None):
        super().__init__(app)
        self.semaphore = asyncio.Semaphore(semaphore_limit)
        self.semaphore_limit = semaphore_limit
        self.paths_to_limit = paths_to_limit or []
        self.logger = logging.getLogger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # If no paths specified or current path is in the list to limit
        if not self.paths_to_limit or any(
            path.startswith(p) for p in self.paths_to_limit
        ):
            # Check if semaphore is full before acquiring it
            if self.semaphore.locked():
                self.logger.warning(
                    f"Semaphore full. Request to {path} is waiting... Limit: {self.semaphore_limit}"
                )

            async with self.semaphore:
                self.logger.debug(f"Acquired semaphore for {path}")
                response = await call_next(request)
                self.logger.debug(f"Released semaphore for {path}")
                return response
        else:
            return await call_next(request)
