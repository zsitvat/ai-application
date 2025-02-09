import logging
import sys

from .logs_json_formatter import JSONFormatter


class LoggerService:

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

    def setup_logger(self, log_level: str = "DEBUG"):
        """Setup the logger with JSON formatting.
        Args:
            log_level (str): The log level string

            Returns:
                logging.Logger: The logger object
        """
        logger = logging.getLogger("logger")
        logger.setLevel(self._get_log_level(log_level))

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(self._get_log_level(log_level))
            handler.setFormatter(JSONFormatter())
            logger.addHandler(handler)

        return logger
