import logging
import json


class JSONFormatter(logging.Formatter):
    """JSON Formatter for logging"""

    def format(self, record):
        """
        Format the log record into JSON

        Args:
            record (LogRecord): The log record object

        Returns:
                str: The formatted log record as JSON
        """

        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "function": record.funcName,
        }
        return json.dumps(log_data)
