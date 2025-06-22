import logging
import os

from schemas.personal_data_filter_schema import PersonalDataFilterConfigSchema
from utils.get_prompt import get_prompt_by_type
from utils.select_model import get_conversation_model


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

        self.logger.info(f"Filtering personal data from text: {text[:50]}...")

        model = await get_conversation_model(
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
            }
        )

        response = await model.invoke(prompt)

        return response.content, text
