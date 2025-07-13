import logging
import os
import time
from datetime import datetime
from typing import Any

import psutil


class SystemService:
    """
    Service for system operations like health check and logs.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # Public methods
    async def health_check(self) -> dict:
        """
        Perform comprehensive application health check.

        Returns:
            dict: Health status information including system metrics
        """
        start_time = time.time()

        try:

            system_health = await self._check_system_metrics()

            env_health = await self._check_environment_variables()

            services_health = await self._check_external_services()

            response_time = round((time.time() - start_time) * 1000, 2)

            overall_status = "healthy"
            if not all(
                [
                    system_health["status"] == "healthy",
                    env_health["status"] == "healthy",
                    services_health["status"] == "healthy",
                ]
            ):
                overall_status = "degraded"

            return {
                "status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "response_time_ms": response_time,
                "version": "0.1.0",
                "system": system_health,
                "environment": env_health,
                "services": services_health,
            }

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "response_time_ms": round((time.time() - start_time) * 1000, 2),
                "error": str(e),
            }

    async def get_logs(
        self,
        level: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        source: str | None = None,
        limit: int = 100,
        page: int = 1,
    ) -> dict:
        """
        Retrieve application logs from log files.

        Args:
            level (str, optional): Log level filter
            start_date (datetime, optional): Start date filter
            end_date (datetime, optional): End date filter
            source (str, optional): Source filter
            limit (int): Maximum number of logs to return
            page (int): Page number for pagination

        Returns:
            dict: Log entries and metadata
        """
        try:
            import json
            from pathlib import Path

            log_file_path = os.getenv("LOG_FILE_PATH")
            if not log_file_path:
                return {
                    "logs": [],
                    "total_count": 0,
                    "page": page,
                    "page_size": limit,
                    "message": "File logging not configured (LOG_FILE_PATH not set)",
                }

            log_file = Path(log_file_path)
            if not log_file.exists():
                return {
                    "logs": [],
                    "total_count": 0,
                    "page": page,
                    "page_size": limit,
                    "message": f"Log file not found: {log_file_path}",
                }

            logs = []

            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            log_entry = json.loads(line)

                            if (
                                level
                                and log_entry.get("level", "").upper() != level.upper()
                            ):
                                continue

                            if (
                                source
                                and source.lower()
                                not in log_entry.get("name", "").lower()
                            ):
                                continue

                            if start_date or end_date:
                                log_timestamp = log_entry.get("timestamp")
                                if log_timestamp:
                                    try:
                                        log_dt = datetime.fromisoformat(
                                            log_timestamp.replace("Z", "+00:00")
                                        )
                                        if start_date and log_dt < start_date:
                                            continue
                                        if end_date and log_dt > end_date:
                                            continue
                                    except (ValueError, TypeError):
                                        pass

                            formatted_entry = {
                                "timestamp": log_entry.get("timestamp"),
                                "level": log_entry.get("level", "INFO"),
                                "message": log_entry.get("message", ""),
                                "source": log_entry.get(
                                    "name", log_entry.get("source")
                                ),
                                "details": {
                                    k: v
                                    for k, v in log_entry.items()
                                    if k
                                    not in [
                                        "timestamp",
                                        "level",
                                        "message",
                                        "name",
                                        "source",
                                    ]
                                }
                                or None,
                            }

                            logs.append(formatted_entry)

                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                self.logger.error(f"Error reading log file: {str(e)}")
                return {
                    "logs": [],
                    "total_count": 0,
                    "page": page,
                    "page_size": limit,
                    "error": f"Error reading log file: {str(e)}",
                }

            logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            total_count = len(logs)
            start_idx = (page - 1) * limit
            end_idx = start_idx + limit
            paginated_logs = logs[start_idx:end_idx]

            return {
                "logs": paginated_logs,
                "total_count": total_count,
                "page": page,
                "page_size": limit,
                "total_pages": (total_count + limit - 1) // limit,
                "has_next": end_idx < total_count,
                "has_previous": page > 1,
            }

        except Exception as e:
            self.logger.error(f"Error retrieving logs: {str(e)}")
            return {
                "logs": [],
                "total_count": 0,
                "page": page,
                "page_size": limit,
                "error": f"Error retrieving logs: {str(e)}",
            }

    # Private methods
    async def _check_system_metrics(self) -> dict[str, Any]:
        """Check system resource metrics."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            cpu_percent = psutil.cpu_percent(interval=1)

            memory_usage = memory.percent
            disk_usage = (disk.used / disk.total) * 100

            status = "healthy"
            issues = []

            if memory_usage > 90:
                status = "unhealthy"
                issues.append(f"High memory usage: {memory_usage:.1f}%")
            elif memory_usage > 80:
                status = "degraded"
                issues.append(f"Elevated memory usage: {memory_usage:.1f}%")

            if disk_usage > 90:
                status = "unhealthy"
                issues.append(f"High disk usage: {disk_usage:.1f}%")
            elif disk_usage > 85:
                status = "degraded"
                issues.append(f"Elevated disk usage: {disk_usage:.1f}%")

            if cpu_percent > 90:
                status = "unhealthy"
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 80:
                status = "degraded"
                issues.append(f"Elevated CPU usage: {cpu_percent:.1f}%")

            return {
                "status": status,
                "metrics": {
                    "memory_usage_percent": round(memory_usage, 2),
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "memory_total_gb": round(memory.total / (1024**3), 2),
                    "disk_usage_percent": round(disk_usage, 2),
                    "disk_free_gb": round(disk.free / (1024**3), 2),
                    "disk_total_gb": round(disk.total / (1024**3), 2),
                    "cpu_usage_percent": round(cpu_percent, 2),
                },
                "issues": issues,
            }

        except Exception as e:
            self.logger.error(f"Error checking system metrics: {str(e)}")
            return {
                "status": "unhealthy",
                "error": f"Failed to get system metrics: {str(e)}",
            }

    async def _check_environment_variables(self) -> dict[str, Any]:
        """Check required environment variables."""
        required_vars = [
            "OPENAI_API_KEY",
            "REDIS_HOST",
            "REDIS_PORT",
            "REDIS_DB",
        ]

        optional_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "REDIS_PASSWORD",
            "LOG_LEVEL",
        ]

        status = "healthy"
        missing = []
        present = []

        for var in required_vars:
            if os.getenv(var):
                present.append(var)
            else:
                missing.append(var)
                status = "unhealthy"

        for var in optional_vars:
            if os.getenv(var):
                present.append(var)

        return {
            "status": status,
            "required_present": len([v for v in required_vars if v in present]),
            "required_total": len(required_vars),
            "optional_present": len([v for v in optional_vars if v in present]),
            "optional_total": len(optional_vars),
            "missing_required": missing,
        }

    async def _check_external_services(self) -> dict[str, Any]:
        """Check connectivity to external services."""
        services = []
        services_status = "healthy"

        # Redis check
        try:
            import redis

            redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                password=os.getenv("REDIS_PASSWORD"),
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            redis_client.ping()
            services.append({"name": "Redis", "status": "healthy"})
        except Exception as e:
            services.append({"name": "Redis", "status": "unhealthy", "error": str(e)})
            services_status = "degraded"

        return {"status": services_status, "services": services}
