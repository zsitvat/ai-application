from langchain import hub
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_core.messages import ToolMessage
from langchain_core.documents import Document
from langchain_core.messages.utils import get_buffer_string

import logging
from typing import Literal
from uuid import uuid4

from utils.model_selector import get_model
from schemas.model_schema import ModelSchema
from schemas.tool_schema import Tool
from .tools_config import AVAILABLE_TOOLS
from .in_memory_history import InMemoryHistory


class AgentService:
    """Service to get answer from the model and tools"""

    def __init__(
        self, prompt: str, model: ModelSchema, tools_config: list[Tool], user_id: str
    ):
        self.prompt = prompt
        self.model = model
        self.user_id = user_id
        self.tools_config = tools_config
        self.tools = self.get_tools()

    async def get_agent_answer(self, user_input: str) -> str:
        """Run the chain to get answer the user input based on the model and tools

        Args:
            user_input (str): User input

        Returns:
            str: Answer from the model
        """
        logging.getLogger("logger").debug("Starting to get AI answer")

        try:

            workflow = await self._get_workflow()

            result = await workflow.ainvoke(
                {"messages": [user_input]},
                config={
                    "configurable": {
                        "thread_id": uuid4(),
                        "tool_config": self.tools_config,
                    }
                },
            )["messages"][-1].content

            history = InMemoryHistory()
            history.add_user_message(user_id=self.user_id, message=user_input)
            history.add_ai_message(user_id=self.user_id, message=result)

            logging.getLogger("logger").debug(f"Agent answer received {result}")

            return result

        except Exception as ex:
            logging.getLogger("logger").error(f"Error while getting agent answer: {ex}")
            raise ex

    def get_tools(self) -> list:
        """Get the tools from the tools config"""

        model_tools = []

        for tool in self.tools_config:
            if tool.name in AVAILABLE_TOOLS:
                model_tools.append(AVAILABLE_TOOLS[tool.name])
            else:
                logging.getLogger("logger").warning(
                    f"Tool {tool.name} not found in available tools."
                )
        return model_tools

    def _get_documents_form_messages(self, messages: list) -> list:
        """Get the documents from the messages"""

        documents = []
        for message in messages:
            if isinstance(message, ToolMessage):
                if isinstance(message.content, Document):
                    documents.append(message.content.page_content)
                elif isinstance(message.content, str):
                    documents.append(message.content)
                elif isinstance(message.content, list):
                    for doc in message.content:
                        if isinstance(doc, Document):
                            documents.append(doc.page_content)
                        elif isinstance(doc, str):
                            documents.append(doc)
        return documents

    async def _call_model(self, state: MessagesState):
        """Call the model to get the answer"""

        messages = state["messages"]
        user_input = messages[0].content
        documents = self._get_documents_form_messages(messages)

        llm = get_model(
            model=self.model.name,
            deployment=self.model.deployment,
            provider=self.model.provider,
            type=self.model.type,
            temperature=self.model.temperature,
        )

        llm = llm.bind_tools(self.tools, tool_choice="auto") if self.tools else llm

        prompt = await hub.pull(self.prompt)

        chain = prompt | llm

        chat_history = get_buffer_string(
           InMemoryHistory().get_messages(user_id=self.user_id)
        )

        answer = await chain.ainvoke(
            {
                "question": user_input,
                "documents": " ".join(doc + "\n\n" for doc in documents),
                "chat_history": chat_history,
            }
        )

        return {"messages": [answer]}

    def _should_continue(
        self, state: MessagesState
    ) -> Literal["tools", "__end__", "agent"]:
        """Check if the agent should continue with the tools or end the conversation"""

        messages = state["messages"]
        last_message = messages[-1]
        for tool_call in last_message.tool_calls:
            self.tools = [tool for tool in self.tools if tool.name != tool_call["name"]]

        if last_message.tool_calls:
            next_node = "tools"
        else:
            next_node = END

        return next_node

    async def _get_workflow(self) -> StateGraph:
        """Get the workflow for the agent service"""

        workflow = StateGraph(MessagesState)

        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.add_edge(START, "agent")

        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
        )

        workflow.add_edge("tools", "agent")

        return await workflow.compile()
