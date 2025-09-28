from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.validators.personal_data.personal_data_filter_checkpointer import (
    PersonalDataFilterCheckpointer,
)


@pytest.mark.asyncio
async def test_aput_filters_personal_data():
    base_checkpointer = AsyncMock()
    personal_data_service = AsyncMock()
    personal_data_service.filter_personal_data.return_value = ("filtered", None)
    config = MagicMock()
    config.config = {"key": "val"}
    logger = MagicMock()
    checkpoint = MagicMock()
    checkpoint.channel_values = {
        "messages": [MagicMock(content="secret")],
        "user_input": "input",
    }
    checkpointer = PersonalDataFilterCheckpointer(
        base_checkpointer, personal_data_service, config, logger
    )
    await checkpointer.aput("cfg", checkpoint, {})
    personal_data_service.filter_personal_data.assert_called()
    base_checkpointer.aput.assert_awaited()


@pytest.mark.asyncio
async def test_aput_no_channel_values():
    base_checkpointer = AsyncMock()
    personal_data_service = AsyncMock()
    config = MagicMock()
    logger = MagicMock()
    checkpoint = MagicMock()
    checkpointer = PersonalDataFilterCheckpointer(
        base_checkpointer, personal_data_service, config, logger
    )
    await checkpointer.aput("cfg", checkpoint, {})
    base_checkpointer.aput.assert_awaited()


@pytest.mark.asyncio
async def test_put_sync():
    base_checkpointer = AsyncMock()
    personal_data_service = AsyncMock()
    config = MagicMock()
    logger = MagicMock()
    checkpoint = MagicMock()
    checkpointer = PersonalDataFilterCheckpointer(
        base_checkpointer, personal_data_service, config, logger
    )
    # Should not raise
    await checkpointer.aput("cfg", checkpoint, {})


def test_getattr_delegation():
    base_checkpointer = MagicMock()
    personal_data_service = MagicMock()
    config = MagicMock()
    logger = MagicMock()
    checkpointer = PersonalDataFilterCheckpointer(
        base_checkpointer, personal_data_service, config, logger
    )
    base_checkpointer.some_method = MagicMock(return_value=42)
    assert checkpointer.some_method() == 42
