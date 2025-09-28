import asyncio
from typing import Optional, Union

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langfuse import Langfuse

from src.config.constants import DEFAULT_CACHE_TTL


async def get_prompt_by_type(
    prompt_id: str, tracer_type: str = "langsmith", cache_ttl: int = DEFAULT_CACHE_TTL
) -> Union[PromptTemplate, ChatPromptTemplate]:
    """Get a prompt template by type and ID.

    Args:
        prompt_id: The ID of the prompt to retrieve
        tracer_type: The type of tracer to use (langfuse or other)
        cache_ttl: Cache time-to-live in seconds

    Returns:
        A prompt template instance

    Raises:
        ValueError: If prompt retrieval fails
    """
    if tracer_type == "langfuse":
        return await _get_langfuse_prompt(prompt_id, cache_ttl)
    else:
        return await asyncio.to_thread(hub.pull, prompt_id)


async def _get_langfuse_prompt(
    prompt_id: str, cache_ttl: int
) -> Union[PromptTemplate, ChatPromptTemplate]:
    """Get a prompt from Langfuse."""
    label: Optional[str] = None
    if ":" in prompt_id:
        prompt_id, label = prompt_id.split(":")

    langfuse = Langfuse()

    if _is_integer(label):
        langfuse_prompt = await asyncio.to_thread(
            langfuse.get_prompt,
            name=prompt_id,
            version=int(label),
            cache_ttl_seconds=cache_ttl,
        )
    else:
        langfuse_prompt = await asyncio.to_thread(
            langfuse.get_prompt,
            name=prompt_id,
            label="latest" if label is None else label,
            cache_ttl_seconds=cache_ttl,
        )

    prompt_type = type(langfuse_prompt)
    if "TextPromptClient" in str(prompt_type):
        return PromptTemplate.from_template(langfuse_prompt.get_langchain_prompt())
    else:
        return ChatPromptTemplate.from_messages(langfuse_prompt.get_langchain_prompt())


def _is_integer(value: Optional[str]) -> bool:
    """Check if a string value can be converted to an integer.

    Args:
        value: The string value to check

    Returns:
        True if the value can be converted to an integer, False otherwise
    """
    if value is None:
        return False

    try:
        int(value)
        return True
    except ValueError:
        return False
