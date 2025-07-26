import json
import logging
from typing import Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from utils.select_model import get_chat_model


class InvalidTopicException(Exception):
    """Exception raised when a topic is invalid."""

    def __init__(self, topic: str, reason: str):
        self.topic = topic
        self.reason = reason
        super().__init__(f"Invalid topic '{topic}': {reason}")


class TopicValidatorService:
    """
    Service for validating if questions belong to predefined topics using LLM.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._default_prompt = """Classify the following text '{text}' into one of these topics: {topics}.
        Format the response as JSON with the following schema: ```json {{"topic": "most_relevant_topic"}}```"""

    async def validate_topic(
        self,
        question: str,
        model_provider: str,
        model_name: str,
        model_deployment: str,
        allowed_topics: Optional[list[str]] = None,
        invalid_topics: Optional[list[str]] = None,
        raise_on_invalid: bool = False,
    ) -> tuple[bool, str, str]:
        """
        Validate if the question belongs to acceptable topics.

        Args:
            question (str): Question to validate
            model_provider (str): LLM provider
            model_name (str): Model name
            model_deployment (str): Model deployment
            valid_topics (list[str], optional): List of valid topics
            invalid_topics (list[str], optional): List of invalid topics
            raise_on_invalid (bool): Whether to raise exception on invalid topic

        Returns:
            tuple[bool, str, str]: (is_valid, topic, reason)

        Raises:
            InvalidTopicException: If raise_on_invalid=True and topic is invalid
        """
        self.logger.info(f"Validating topic for question: {question[:50]}...")

        if invalid_topics is None:
            invalid_topics = [
                "personal",
                "politics",
                "religion",
                "inappropriate",
                "programming",
                "other",
            ]

        if not allowed_topics:
            raise ValueError(
                "allowed_topics must be set and contain at least one topic."
            )

        allowed_set = set(allowed_topics)
        invalid_set = set(invalid_topics)

        if allowed_set.intersection(invalid_set):
            raise ValueError("A topic cannot be allowed and invalid at the same time.")

        if "other" not in invalid_set and "other" not in allowed_set:
            invalid_set.add("other")

        candidate_topics = list(allowed_set.union(invalid_set))

        try:
            topic = await self._classify_with_llm(
                question, candidate_topics, model_provider, model_name, model_deployment
            )

            is_allowed = topic in allowed_topics

            if is_allowed:
                reason = f"Question classified as '{topic}' which is an allowed topic."
                self.logger.debug(f"Topic validation passed: {topic}")
                return True, topic, reason
            else:
                reason = (
                    f"Question classified as '{topic}' which is not an allowed topic."
                )
                self.logger.debug(f"Topic validation failed: {topic}")

                if raise_on_invalid:
                    raise InvalidTopicException(topic, reason)

                return False, topic, reason

        except Exception as ex:
            error_msg = f"Error during topic validation: {str(ex)}"
            self.logger.error(error_msg)

            if raise_on_invalid and isinstance(ex, InvalidTopicException):
                raise

            return False, "error", error_msg

    async def _classify_with_llm(
        self,
        text: str,
        topics: list[str],
        model_provider: str,
        model_name: str,
        model_deployment: str,
    ) -> str:
        """
        Classify text using LLM.

        Args:
            text (str): Text to classify
            topics (list[str]): List of candidate topics
            model_provider (str): LLM provider
            model_name (str): Model name
            model_deployment (str): Model deployment

        Returns:
            str: Classified topic
        """
        try:
            llm = get_chat_model(
                provider=model_provider,
                deployment=model_deployment,
                model=model_name,
            )

            prompt = ChatPromptTemplate.from_template(self._default_prompt)

            chain = prompt | llm | JsonOutputParser()

            response = await chain.ainvoke({"text": text, "topics": ", ".join(topics)})

            if isinstance(response, dict) and "topic" in response:
                return response["topic"]
            else:
                self.logger.warning(f"Unexpected LLM response format: {response}")
                return "other"

        except json.JSONDecodeError as e:
            self.logger.warning(f"Cannot parse LLM result, not a JSON format: {e}")
            return "other"
        except Exception as e:
            self.logger.error(f"Error calling LLM for topic classification: {e}")
            raise
