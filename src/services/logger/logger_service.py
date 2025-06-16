import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .logs_json_formatter import JSONFormatter


class LoggerService:
    def __init__(self):
        self._loggers = {}

    def _get_log_level(self, log_level: str = "DEBUG"):
        """Get the log level based on the log level string.

        Args:
            log_level (str): The log level string

        Returns:
            int: The log level integer
        """
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
            "FATAL": logging.FATAL,
            "NOTSET": logging.NOTSET,
        }
        return log_levels.get(log_level.upper(), logging.DEBUG)

    def _ensure_log_directory(self, log_file_path: str):
        """Ensure the log directory exists."""
        log_dir = Path(log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)

    def setup_logger(self, log_level: str = "DEBUG", logger_name: str = "logger"):
        """Setup the logger with JSON formatting for both console and file output.

        Args:
            log_level (str): The log level string
            logger_name (str): Name of the logger

        Returns:
            logging.Logger: The logger object
        """
        if logger_name in self._loggers:
            return self._loggers[logger_name]

        logger = logging.getLogger(logger_name)
        logger.setLevel(self._get_log_level(log_level))

        logger.handlers.clear()
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self._get_log_level(log_level))
        console_handler.setFormatter(JSONFormatter())
        logger.addHandler(console_handler)

        log_file_path = os.getenv("LOG_FILE_PATH")
        if log_file_path:
            try:
                self._ensure_log_directory(log_file_path)

                max_bytes = (
                    int(os.getenv("LOG_MAX_FILE_SIZE_MB", "10")) * 1024 * 1024
                )  # Default 10MB
                backup_count = int(
                    os.getenv("LOG_BACKUP_COUNT", "5")
                )  # Default 5 backup files

                file_handler = RotatingFileHandler(
                    log_file_path,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                file_handler.setLevel(self._get_log_level(log_level))
                file_handler.setFormatter(JSONFormatter())
                logger.addHandler(file_handler)

                logger.info(f"File logging enabled: {log_file_path}")

            except Exception as e:
                logger.error(f"Failed to setup file logging: {str(e)}")

        logger.propagate = False

        self._loggers[logger_name] = logger
        return logger

    def get_logger(self, logger_name: str = "logger"):
        """Get an existing logger or create a new one.

        Args:
            logger_name (str): Name of the logger

        Returns:
            logging.Logger: The logger object
        """
        if logger_name in self._loggers:
            return self._loggers[logger_name]

        # Create logger with default settings if it doesn't exist
        return self.setup_logger(logger_name=logger_name)

    def update_log_level(self, log_level: str, logger_name: str = "logger"):
        """Update the log level for an existing logger.

        Args:
            log_level (str): The new log level
            logger_name (str): Name of the logger
        """
        if logger_name in self._loggers:
            logger = self._loggers[logger_name]
            new_level = self._get_log_level(log_level)
            logger.setLevel(new_level)

            # Update all handlers
            for handler in logger.handlers:
                handler.setLevel(new_level)

            logger.info(f"Log level updated to {log_level}")

    def list_log_files(self):
        """List all log files in the log directory.

        Returns:
            list: List of log file paths
        """
        log_file_path = os.getenv("LOG_FILE_PATH")
        if not log_file_path:
            return []

        log_dir = Path(log_file_path).parent
        if not log_dir.exists():
            return []

        log_files = []
        base_name = Path(log_file_path).name

        for file_path in log_dir.glob(f"{base_name}*"):
            if file_path.is_file():
                log_files.append(str(file_path))

        return sorted(log_files)
