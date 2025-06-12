from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langfuse import Langfuse
import asyncio


async def get_prompt_by_type(prompt_id, tracer_type, cache_ttl=60):

    if tracer_type == "langfuse":

        label = None
        if ":" in prompt_id:
            prompt_id, label = prompt_id.split(":")

        if await is_integer(label):
            langfuse_prompt = await asyncio.to_thread(
                Langfuse().get_prompt,
                name=prompt_id,
                version=int(label),
                cache_ttl_seconds=cache_ttl,
            )
        else:
            langfuse_prompt = await asyncio.to_thread(
                Langfuse().get_prompt,
                name=prompt_id,
                label="latest" if label is None else label,
                cache_ttl_seconds=cache_ttl,
            )

        prompt_type = type(langfuse_prompt)
        if "TextPromptClient" in str(prompt_type):
            return PromptTemplate.from_template(langfuse_prompt.get_langchain_prompt())
        else:
            return ChatPromptTemplate.from_messages(
                langfuse_prompt.get_langchain_prompt()
            )

    else:
        return await asyncio.to_thread(hub.pull, prompt_id)


async def is_integer(value):
    try:
        if value is None:
            return False
        else:
            int(value)
            return True
    except ValueError:
        return False
