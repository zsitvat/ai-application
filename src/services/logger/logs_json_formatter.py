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
            "module": record.module,
            "line": record.lineno,
            "filename": record.filename,
            "name": record.name,
            "process": record.process,
            "thread": record.thread,
            "thread_name": record.threadName,
            "logger": record.name,
            "stack_info": self.formatStack(record.stack_info),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)
