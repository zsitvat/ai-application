import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import psutil
import redis

from src.config.constants import (
    DEFAULT_LIMIT,
    DEFAULT_PAGE,
    HEALTH_STATUS_HEALTHY,
    HEALTH_STATUS_UNHEALTHY,
)


class SystemService:
    """
    Service for system operations like health check and logs.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _check_system_metrics(self) -> dict:
        """Check system resource metrics."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            cpu_percent = psutil.cpu_percent(interval=1)

            memory_usage = memory.percent
            disk_usage = (disk.used / disk.total) * 100

            status = HEALTH_STATUS_HEALTHY
            issues = []

            if memory_usage > 80:
                status = HEALTH_STATUS_UNHEALTHY
                issues.append(f"High memory usage: {memory_usage:.1f}%")

            if disk_usage > 85:
                status = HEALTH_STATUS_UNHEALTHY
                issues.append(f"High disk usage: {disk_usage:.1f}%")

            if cpu_percent > 80:
                status = HEALTH_STATUS_UNHEALTHY
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")

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
            return {
                "status": HEALTH_STATUS_UNHEALTHY,
                "error": f"Failed to get system metrics: {str(e)}",
            }

    def _check_environment_variables(self) -> dict:
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

        status = HEALTH_STATUS_HEALTHY
        missing = []
        present = []

        for var in required_vars:
            if os.getenv(var):
                present.append(var)
            else:
                missing.append(var)
                status = HEALTH_STATUS_UNHEALTHY

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

    def _check_external_services(self) -> dict:
        """Check connectivity to external services."""
        services = []
        services_status = HEALTH_STATUS_HEALTHY
        try:

            redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                password=os.getenv("REDIS_PASSWORD"),
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            redis_client.ping()
            services.append({"name": "Redis", "status": HEALTH_STATUS_HEALTHY})
        except Exception as e:
            services.append(
                {"name": "Redis", "status": HEALTH_STATUS_UNHEALTHY, "error": str(e)}
            )
            services_status = HEALTH_STATUS_UNHEALTHY

        return {"status": services_status, "services": services}

    def _calculate_overall_status(
        self, system_health: dict, env_health: dict, services_health: dict
    ) -> str:
        """Calculate overall health status based on individual checks."""
        if all(
            check["status"] == HEALTH_STATUS_HEALTHY
            for check in [system_health, env_health, services_health]
        ):
            return HEALTH_STATUS_HEALTHY
        return HEALTH_STATUS_UNHEALTHY

    def _build_health_response(
        self,
        status: str,
        start_time: float,
        system_health: dict = None,
        env_health: dict = None,
        services_health: dict = None,
        error: str = None,
    ) -> dict:
        """Build health check response dictionary."""
        response_time = round((time.time() - start_time) * 1000, 2)
        base_response = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": response_time,
            "version": "0.1.0",
        }

        if error:
            base_response["error"] = error
        else:
            base_response.update(
                {
                    "system": system_health,
                    "environment": env_health,
                    "services": services_health,
                }
            )

        return base_response

    async def health_check(self) -> dict:
        """Perform comprehensive application health check.

        Returns:
            dict: Health status information including system metrics
        """
        self.logger.info("[SystemService|health_check] started")
        start_time = time.time()

        try:
            system_health = await asyncio.to_thread(self._check_system_metrics)
            env_health = self._check_environment_variables()
            services_health = await asyncio.to_thread(self._check_external_services)

            overall_status = self._calculate_overall_status(
                system_health, env_health, services_health
            )

            self.logger.info("[SystemService|health_check] finished")
            return self._build_health_response(
                overall_status, start_time, system_health, env_health, services_health
            )
        except Exception as e:
            self.logger.error(f"[SystemService] Health check failed: {str(e)}")
            self.logger.info("[SystemService|health_check] finished")
            return self._build_health_response(
                HEALTH_STATUS_UNHEALTHY, start_time, error=str(e)
            )

    async def get_logs(
        self,
        level: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        source: str | None = None,
        limit: int = DEFAULT_LIMIT,
        page: int = DEFAULT_PAGE,
    ) -> dict:
        """Retrieve application logs from log files.

        Args:
            level: Log level filter
            start_date: Start date filter
            end_date: End date filter
            source: Source filter
            limit: Maximum number of logs to return
            page: Page number for pagination

        Returns:
            dict: Log entries and metadata
        """
        self.logger.info("[SystemService|get_logs] started")

        try:
            log_file_path = os.getenv("LOG_FILE_PATH")
            validation_result = self._validate_log_file_path(log_file_path, page, limit)
            if validation_result:
                return validation_result

            logs = await asyncio.to_thread(
                self._read_and_filter_logs,
                log_file_path,
                level,
                start_date,
                end_date,
                source,
            )
            if isinstance(logs, dict):
                return logs

            paginated_result = self._paginate_logs(logs, page, limit)
            self.logger.info("[SystemService|get_logs] finished")
            return paginated_result

        except Exception as e:
            self.logger.error(f"[SystemService] Error retrieving logs: {str(e)}")
            self.logger.info("[SystemService|get_logs] finished")
            return self._build_error_response(
                page, limit, f"Error retrieving logs: {str(e)}"
            )

    def _validate_log_file_path(
        self, log_file_path: str | None, page: int, limit: int
    ) -> dict | None:
        """Validate log file path and existence."""
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
        return None

    def _read_and_filter_logs(
        self,
        log_file_path: str,
        level: str | None,
        start_date: datetime | None,
        end_date: datetime | None,
        source: str | None,
    ) -> list[dict] | dict:
        """Read and filter log entries from file."""

        logs = []
        log_file = Path(log_file_path)

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        log_entry = json.loads(line)
                        if self._should_include_log_entry(
                            log_entry, level, start_date, end_date, source
                        ):
                            formatted_entry = self._format_log_entry(log_entry)
                            logs.append(formatted_entry)
                    except json.JSONDecodeError:
                        continue
            return logs
        except Exception as e:
            self.logger.error(f"[SystemService] Error reading log file: {str(e)}")
            return self._build_error_response(0, 0, f"Error reading log file: {str(e)}")

    def _should_include_log_entry(
        self,
        log_entry: dict,
        level: str | None,
        start_date: datetime | None,
        end_date: datetime | None,
        source: str | None,
    ) -> bool:
        """Check if log entry should be included based on filters."""
        if not self._passes_level_filter(log_entry, level):
            return False

        if not self._passes_source_filter(log_entry, source):
            return False

        if not self._passes_date_filter(log_entry, start_date, end_date):
            return False

        return True

    def _passes_level_filter(self, log_entry: dict, level: str | None) -> bool:
        """Check if log entry passes level filter."""
        return not level or log_entry.get("level", "").upper() == level.upper()

    def _passes_source_filter(self, log_entry: dict, source: str | None) -> bool:
        """Check if log entry passes source filter."""
        return not source or source.lower() in log_entry.get("name", "").lower()

    def _passes_date_filter(
        self, log_entry: dict, start_date: datetime | None, end_date: datetime | None
    ) -> bool:
        """Check if log entry passes date filter."""
        if not (start_date or end_date):
            return True

        log_timestamp = log_entry.get("timestamp")
        if not log_timestamp:
            return True

        try:
            log_dt = datetime.fromisoformat(log_timestamp.replace("Z", "+00:00"))
            if start_date and log_dt < start_date:
                return False
            if end_date and log_dt > end_date:
                return False
        except (ValueError, TypeError):
            pass

        return True

    def _format_log_entry(self, log_entry: dict) -> dict:
        """Format a log entry for response."""
        excluded_keys = {"timestamp", "level", "message", "name", "source"}
        details = {k: v for k, v in log_entry.items() if k not in excluded_keys}

        return {
            "timestamp": log_entry.get("timestamp"),
            "level": log_entry.get("level", "INFO"),
            "message": log_entry.get("message", ""),
            "source": log_entry.get("name", log_entry.get("source")),
            "details": details if details else None,
        }

    def _paginate_logs(self, logs: list[dict], page: int, limit: int) -> dict:
        """Paginate log entries."""
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

    def _build_error_response(self, page: int, limit: int, error_message: str) -> dict:
        """Build error response for log operations."""
        return {
            "logs": [],
            "total_count": 0,
            "page": page,
            "page_size": limit,
            "error": error_message,
        }
