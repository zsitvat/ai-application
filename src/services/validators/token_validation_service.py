import logging

from src.schemas.token_validation_schema import (
    TokenCountResult,
    TokenEstimationResult,
    TruncationResult,
)
from src.utils.token_counter import token_counter


class TokenValidationService:
    """Service for validating and truncating text based on token limits."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def validate_and_truncate_text(
        self,
        text: str,
        max_tokens: int,
        encoding_name: str = "cl100k_base",
        truncate_from_end: bool = True,
        preserve_sentences: bool = False,
    ) -> TruncationResult:
        """
        Validate text token count and truncate if necessary.

        Args:
            text (str): Input text to validate and potentially truncate
            max_tokens (int): Maximum allowed tokens
            encoding_name (str): Encoding name for token counting (default: "cl100k_base")
            truncate_from_end (bool): If True, truncate from end; if False, from beginning
            preserve_sentences (bool): If True, try to preserve complete sentences

        Returns:
            TruncationResult: Contains original_text, truncated_text, original_tokens, final_tokens, was_truncated
        """
        try:
            original_token_count = await token_counter(text, encoding_name)

            self.logger.debug(
                f"Original text has {original_token_count} tokens, limit is {max_tokens}"
            )

            if original_token_count <= max_tokens:
                return TruncationResult(
                    original_text=text,
                    truncated_text=text,
                    original_tokens=original_token_count,
                    final_tokens=original_token_count,
                    was_truncated=False,
                    encoding_used=encoding_name,
                )

            self.logger.info(
                f"Text exceeds token limit ({original_token_count} > {max_tokens}), truncating..."
            )

            truncated_text = await self._truncate_text(
                text, max_tokens, encoding_name, truncate_from_end, preserve_sentences
            )

            final_token_count = await token_counter(truncated_text, encoding_name)

            return TruncationResult(
                original_text=text,
                truncated_text=truncated_text,
                original_tokens=original_token_count,
                final_tokens=final_token_count,
                was_truncated=True,
                encoding_used=encoding_name,
            )

        except Exception as e:
            self.logger.error(f"Error in token validation: {str(e)}")
            raise e

    async def _truncate_text(
        self,
        text: str,
        max_tokens: int,
        encoding_name: str,
        truncate_from_end: bool,
        preserve_sentences: bool,
    ) -> str:
        """Truncate text to fit within token limit."""
        if preserve_sentences:
            return await self._truncate_preserving_sentences(
                text, max_tokens, encoding_name, truncate_from_end
            )
        else:
            return await self._truncate_by_characters(
                text, max_tokens, encoding_name, truncate_from_end
            )

    async def _truncate_preserving_sentences(
        self,
        text: str,
        max_tokens: int,
        encoding_name: str,
        truncate_from_end: bool,
    ) -> str:
        """Truncate text while preserving complete sentences."""
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        if not sentences:
            return await self._truncate_by_characters(
                text, max_tokens, encoding_name, truncate_from_end
            )

        if truncate_from_end:
            result_sentences = []
            current_text = ""

            for sentence in sentences:
                test_text = current_text + sentence + ". "
                test_tokens = await token_counter(test_text.strip(), encoding_name)

                if test_tokens <= max_tokens:
                    result_sentences.append(sentence)
                    current_text = test_text
                else:
                    break

            return ". ".join(result_sentences) + "." if result_sentences else ""
        else:
            result_sentences = []
            current_text = ""

            for sentence in reversed(sentences):
                test_text = sentence + ". " + current_text
                test_tokens = await token_counter(test_text.strip(), encoding_name)

                if test_tokens <= max_tokens:
                    result_sentences.insert(0, sentence)
                    current_text = test_text
                else:
                    break

            return ". ".join(result_sentences) + "." if result_sentences else ""

    async def _truncate_by_characters(
        self,
        text: str,
        max_tokens: int,
        encoding_name: str,
        truncate_from_end: bool,
    ) -> str:
        """Truncate text by progressively removing characters until token count fits."""
        estimated_chars_per_token = len(text) / await token_counter(text, encoding_name)
        estimated_max_chars = int(max_tokens * estimated_chars_per_token * 0.9)

        if truncate_from_end:
            truncated = text[:estimated_max_chars]
        else:
            truncated = text[-estimated_max_chars:]

        while len(truncated) > 0:
            token_count = await token_counter(truncated, encoding_name)
            if token_count <= max_tokens:
                break

            chars_to_remove = max(1, len(truncated) // 20)

            if truncate_from_end:
                truncated = truncated[:-chars_to_remove]
            else:
                truncated = truncated[chars_to_remove:]

        return truncated

    async def check_token_count(
        self, text: str, encoding_name: str = "cl100k_base"
    ) -> TokenCountResult:
        """
        Check token count of text without truncating.

        Args:
            text (str): Text to analyze
            encoding_name (str): Encoding name for token counting

        Returns:
            TokenCountResult: Contains text, token_count, character_count, encoding_used
        """
        try:
            token_count = await token_counter(text, encoding_name)

            return TokenCountResult(
                text=text,
                token_count=token_count,
                character_count=len(text),
                encoding_used=encoding_name,
                chars_per_token=len(text) / token_count if token_count > 0 else 0,
            )

        except Exception as e:
            self.logger.error(f"Error checking token count: {str(e)}")
            raise e

    async def estimate_tokens_for_text_length(
        self,
        target_tokens: int,
        encoding_name: str = "cl100k_base",
        sample_text: str | None = None,
    ) -> TokenEstimationResult:
        """
        Estimate character count needed for target token count.

        Args:
            target_tokens (int): Desired token count
            encoding_name (str): Encoding name
            sample_text (str, optional): Sample text to estimate ratio

        Returns:
            TokenEstimationResult: Contains estimated_characters, target_tokens, encoding_used
        """
        try:
            if sample_text:
                sample_tokens = await token_counter(sample_text, encoding_name)
                chars_per_token = (
                    len(sample_text) / sample_tokens if sample_tokens > 0 else 4
                )
            else:
                chars_per_token = 4

            estimated_chars = int(target_tokens * chars_per_token)

            return TokenEstimationResult(
                estimated_characters=estimated_chars,
                target_tokens=target_tokens,
                encoding_used=encoding_name,
                chars_per_token_ratio=chars_per_token,
            )

        except Exception as e:
            self.logger.error(f"Error estimating text length: {str(e)}")
            raise e
