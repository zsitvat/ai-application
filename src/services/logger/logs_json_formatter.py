import json
import logging


class JSONFormatter(logging.Formatter):
    """JSON Formatter for loggin[A-Z]"""

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
