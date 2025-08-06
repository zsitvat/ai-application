import logging
import os

from src.schemas.personal_data_filter_schema import PersonalDataFilterConfigSchema
from src.utils.get_prompt import get_prompt_by_type
from src.utils.select_model import get_chat_model


class PersonalDataFilterService:
    """
    Service for filtering personal and sensitive data from text.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def filter_personal_data(
        self, text: str, config: PersonalDataFilterConfigSchema
    ) -> tuple[str, str]:
        """
        Filter personal data from text using AI.

        Args:
            text (str): Original text to filter
        Returns:
            tuple[str, str]: (filtered_text, original_text)
        """

        self.logger.info("Filtering personal data from text...")

        model = get_chat_model(
            provider=config.model.provider,
            model=config.model.name,
            deployment=config.model.deployment,
        )

        prompt = await get_prompt_by_type(
            config.prompt,
            tracer_type=os.getenv("TRACER_TYPE", "langsmith"),
            cache_ttl=os.getenv("CACHE_TTL", "60"),
        )

        prompt.append(
            {
                "role": "user",
                "content": text,
                "mask_char": getattr(config, "mask_char", "*"),
            }
        )

        response = await model.invoke(prompt)

        return response.content, text
