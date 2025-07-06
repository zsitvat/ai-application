import logging


class TopicValidatorService:
    """
    Service for validating if questions belong to predefined topics using LLM.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def validate_topic(self, question: str) -> tuple[bool, str, str]:
        """
        Validate if the question belongs to acceptable topics.

        Args:
            question (str): Question to validate
            user_id (str): User identifier

        Returns:
            tuple[bool, str, str]: (is_valid, topic, reason)
        """
        # TODO: Implement LLM-based topic validation
        # Should use predefined topic selection prompt
        self.logger.info(f"Validating topic for question: {question[:50]}...")
        raise NotImplementedError("Topic validation not implemented yet")
