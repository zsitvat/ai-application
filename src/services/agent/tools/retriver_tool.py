from langchain_core.tools import tool
from langchain_community.vectorstores.deeplake import DeepLake
from langchain.embeddings.base import Embeddings
from langchain_core.runnables.config import RunnableConfig

import logging

from utils.model_selector import get_model
from schemas.tool_schema import RetriverTool


@tool
async def retriver_tool(config: RunnableConfig, user_input: str) -> list:
    """
    Web search tool used to search the web for information.

    Args:
        tool_config (dict): The parameters used for the retrieval.
        user_input (str): The user input used to retrieve the information.

    Returns:
        list: A list of documents retrieved from the web.
    """

    try:
        tool_config: RetriverTool = next(
            (
                tool
                for tool in config.get("configurable", {}).get("tool_config", [])
                if isinstance(tool, RetriverTool)
            ),
            None,
        )

        embedding = get_model(
            provider=tool_config.model.provider,
            deployment=tool_config.model.deployment,
            type="embedding",
            model=tool_config.model.name,
        )

        if not isinstance(embedding, Embeddings):
            raise TypeError(
                "Expected an instance of Embeddings. You must provide embedding model. You provided: ",
                type(embedding),
            )

        retriver = DeepLake(
            embedding=embedding, dataset_path=tool_config.vector_db_path, read_only=True
        ).as_retriever(
            search_type=tool_config.search_kwargs.search_type,
            k=tool_config.search_kwargs.k,
            threshold=tool_config.search_kwargs.threshold,
        )

        docs = await retriver.ainvoke(user_input)

        return [doc.page_content for doc in docs]

    except Exception as ex:
        logging.getLogger("logger").error(f"Error in retriver_tool: {ex}")
        raise ex
