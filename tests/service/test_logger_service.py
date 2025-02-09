from src.services.logger.logger_service import LoggerService

import logging
from io import StringIO


async def test_logger_service_output():
    log_output = StringIO()
    handler = logging.StreamHandler(log_output)
    logger_service = LoggerService()
    logger = await logger_service.setup_logger("DEBUG")
    logger.addHandler(handler)

    logger.debug("test")
    handler.flush()
    assert "test" in log_output.getvalue()
