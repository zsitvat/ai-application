from datetime import datetime
import logging
import time
import psutil
import os
from typing import Dict, Any


class SystemService:
    """
    Service for system operations like health check and logs.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

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

    async def _check_system_metrics(self) -> Dict[str, Any]:
        """Check system resource metrics."""
        try:
            memory = psutil.virtual_memory()
            memory_usage = {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent_used": memory.percent,
                "status": "healthy" if memory.percent < 85 else "warning",
            }

            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_usage = {
                "percent_used": cpu_percent,
                "core_count": psutil.cpu_count(),
                "status": "healthy" if cpu_percent < 80 else "warning",
            }

            disk = psutil.disk_usage("/")
            disk_usage = {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent_used": round((disk.used / disk.total) * 100, 2),
                "status": (
                    "healthy" if (disk.used / disk.total) * 100 < 85 else "warning"
                ),
            }

            system_status = "healthy"
            if any(
                metric["status"] == "warning"
                for metric in [memory_usage, cpu_usage, disk_usage]
            ):
                system_status = "warning"

            return {
                "status": system_status,
                "memory": memory_usage,
                "cpu": cpu_usage,
                "disk": disk_usage,
                "uptime_seconds": round(time.time() - psutil.boot_time(), 2),
            }

        except Exception as e:
            self.logger.error(f"System metrics check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}

    async def _check_environment_variables(self) -> Dict[str, Any]:
        """Check required environment variables."""
        required_vars = ["OPENAI_API_KEY", "LANGCHAIN_API_KEY", "SERPAPI_API_KEY"]

        optional_vars = [
            "AZURE_OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "LANGFUSE_SECRET_KEY",
        ]

        missing_required = []
        missing_optional = []

        for var in required_vars:
            if not os.getenv(var):
                missing_required.append(var)

        for var in optional_vars:
            if not os.getenv(var):
                missing_optional.append(var)

        status = "healthy" if not missing_required else "unhealthy"

        return {
            "status": status,
            "required_vars_present": len(required_vars) - len(missing_required),
            "optional_vars_present": len(optional_vars) - len(missing_optional),
            "missing_required": missing_required,
            "missing_optional": missing_optional,
        }

    async def _check_external_services(self) -> Dict[str, Any]:
        """Check external service connectivity."""
        services = {}

        tracer_type = os.getenv("TRACER_TYPE", "").lower()

        if tracer_type == "langfuse":
            services["langfuse"] = {
                "status": "healthy" if os.getenv("LANGFUSE_SECRET_KEY") else "warning",
                "message": (
                    "API key configured"
                    if os.getenv("LANGFUSE_SECRET_KEY")
                    else "API key missing"
                ),
            }
        elif tracer_type == "langsmith":
            services["langsmith"] = {
                "status": "healthy" if os.getenv("LANGCHAIN_API_KEY") else "warning",
                "message": (
                    "API key configured"
                    if os.getenv("LANGCHAIN_API_KEY")
                    else "API key missing"
                ),
            }
        else:
            services["tracer"] = {
                "status": "warning",
                "message": "No tracer service configured (TRACER_TYPE not set)",
            }

        services_status = "healthy"
        if any(service["status"] == "unhealthy" for service in services.values()):
            services_status = "unhealthy"
        elif any(service["status"] == "warning" for service in services.values()):
            services_status = "warning"

        return {"status": services_status, "services": services}

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
                    "limit": limit,
                    "message": "File logging not configured (LOG_FILE_PATH not set)",
                }

            log_file = Path(log_file_path)
            if not log_file.exists():
                return {
                    "logs": [],
                    "total_count": 0,
                    "page": page,
                    "limit": limit,
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

                            logs.append(log_entry)

                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                self.logger.error(f"Error reading log file: {str(e)}")
                return {
                    "logs": [],
                    "total_count": 0,
                    "page": page,
                    "limit": limit,
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
                "limit": limit,
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
                "limit": limit,
                "error": f"Error retrieving logs: {str(e)}",
            }
