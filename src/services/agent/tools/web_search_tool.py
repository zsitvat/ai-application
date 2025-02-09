from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.runnables.config import RunnableConfig

import logging

from schemas.tool_schema import WebSearchTool


@tool
async def web_search_tool(config: RunnableConfig, user_input: str) -> list | str:
    """
    Web search tool is used to search the web for information

    Args:
        tool_config (dict): The parameters to be used for the retrival
        user_input (str): The user input to be used to retrive the documents

    Returns:
        list: The list of documents retrived from the document database
    """

    try:
        tool_config: WebSearchTool = next(
            (
                tool
                for tool in config.get("configurable", {}).get("tool_config", [])
                if isinstance(tool, WebSearchTool)
            ),
            None,
        )

        search = SerpAPIWrapper(params={"engine": tool_config.engine})

        search_results = await search.arun(user_input)

        return search_results

    except Exception as ex:
        logging.getLogger("logger").error(f"Error in web_search_tool: {ex}")
        raise ex
