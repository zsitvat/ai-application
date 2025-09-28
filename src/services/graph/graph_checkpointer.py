import importlib

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.redis import AsyncRedisSaver

from src.schemas.graph_schema import CheckpointerType
from src.services.chat_history.redis_chat_history import RedisChatHistoryService
from src.services.data_api.chat_history import DataChatHistoryService
from src.services.validators.personal_data.personal_data_filter_checkpointer import (
    PersonalDataFilterCheckpointer,
)
from src.services.validators.personal_data.personal_data_filter_service import (
    PersonalDataFilterService,
)


async def create_checkpointer(
    graph_config,
    logger,
    redis_url,
    find_personal_data_filter_config,
    async_redis_saver=None,
):
    """Create checkpointer based on configuration (async version)."""
    if graph_config is None:
        raise ValueError(
            "Graph configuration is not loaded. Cannot create checkpointer."
        )

    checkpointer_type = graph_config.checkpointer_type
    base_checkpointer = None

    if checkpointer_type == CheckpointerType.MEMORY:
        base_checkpointer = InMemorySaver()
    elif checkpointer_type == CheckpointerType.REDIS:
        try:
            if async_redis_saver is None:
                async with AsyncRedisSaver.from_conn_string(
                    redis_url=redis_url
                ) as saver:
                    await saver.asetup()
                    async_redis_saver = saver
            base_checkpointer = async_redis_saver
        except Exception as ex:
            logger.warning(
                f"[GraphService] Failed to create AsyncRedisSaver: {ex}, falling back to memory"
            )
            base_checkpointer = InMemorySaver()
    elif checkpointer_type == CheckpointerType.DATA:
        base_checkpointer = DataChatHistoryService()
    elif checkpointer_type == CheckpointerType.CUSTOM:
        if hasattr(graph_config, "custom_checkpointer_class"):
            module_path, class_name = graph_config.custom_checkpointer_class.rsplit(
                ".", 1
            )
            module = importlib.import_module(module_path)
            checkpointer_class = getattr(module, class_name)
            base_checkpointer = checkpointer_class()
        else:
            base_checkpointer = RedisChatHistoryService()
    else:
        logger.warning(
            f"[GraphService] Unknown checkpointer type: {checkpointer_type}, falling back to memory"
        )
        base_checkpointer = InMemorySaver()

    # Wrap with personal data filter if configured
    personal_data_config = find_personal_data_filter_config()
    if personal_data_config:
        personal_data_service = PersonalDataFilterService()
        return PersonalDataFilterCheckpointer(
            base_checkpointer=base_checkpointer,
            personal_data_service=personal_data_service,
            personal_data_config=personal_data_config,
            logger=logger,
        )

    return base_checkpointer
