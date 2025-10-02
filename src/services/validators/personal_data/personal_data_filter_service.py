import logging
import re

from schemas.model_schema import Model
from src.utils.get_prompt import get_prompt_by_type
from src.utils.select_model import get_model


class PersonalDataFilterService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def apply_regex_replacements(
        self, text: str, regex_patterns: list[str], mask_char: str = "*"
    ) -> str:
        filtered_text = text

        for pattern in regex_patterns:
            try:

                def replace_match(match):
                    return mask_char * len(match.group())

                filtered_text = re.sub(pattern, replace_match, filtered_text)
                self.logger.info(
                    f"[PersonalDataFilterService] Applied regex pattern: {pattern}"
                )
            except re.error as e:
                self.logger.error(
                    f"[PersonalDataFilterService] Invalid regex pattern '{pattern}': {e}"
                )

        return filtered_text

    async def filter_personal_data(
        self,
        text: str,
        model: Model,
        sensitive_words: list[str] = None,
        regex_patterns: list[str] = None,
        prompt: str = None,
        mask_char: str = "*",
    ) -> str:
        if not text or not text.strip():
            self.logger.warning("[PersonalDataFilterService] Empty text provided")
            return text

        try:
            filtered_text = text

            if regex_patterns:
                filtered_text = self.apply_regex_replacements(
                    filtered_text, regex_patterns, mask_char
                )
                self.logger.info(
                    f"[PersonalDataFilterService] Applied {len(regex_patterns)} regex patterns"
                )

            if prompt:
                chat_model = get_model(
                    model.provider,
                    model.deployment,
                    model=getattr(model, "name", None),
                    type="chat",
                )
                prompt_template = await get_prompt_by_type(prompt)

                messages = prompt_template.format_messages(
                    text=filtered_text,
                    mask_char=mask_char,
                    sensitive_words=sensitive_words or [],
                    regex_patterns=regex_patterns or [],
                )

                response = chat_model.invoke(messages)
                filtered_text = response.content
                self.logger.info(
                    "[PersonalDataFilterService] AI filtering completed successfully"
                )
            else:
                self.logger.info(
                    "[PersonalDataFilterService] No prompt provided, skipping AI filtering"
                )

            self.logger.info(
                "[PersonalDataFilterService] Personal data filtering completed successfully"
            )
            return filtered_text

        except Exception as e:
            self.logger.error(
                f"[PersonalDataFilterService] Error filtering personal data: {e}"
            )
            raise e
