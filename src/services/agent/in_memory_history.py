from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

import schedule
from threading import Thread
from datetime import datetime

from config import Config

store = {}


class InMemoryHistory:
    """
    A class to manage in-memory chat history for different users.
    """

    async def get_history(self, user_id: str) -> BaseChatMessageHistory:
        """
        Retrieve the chat history for a given user.
        If the history does not exist, create a new one.

        Args:
            user_id (str): The unique identifier for the user.

        Returns:
            BaseChatMessageHistory: The chat history associated with the user.
        """
        if user_id not in store:
            store[user_id] = ChatMessageHistory()
        return store[user_id]

    async def get_messages(self, user_id: str) -> list:
        """
        Retrieve all messages for a given user.

        Args:
            user_id (str): The unique identifier for the user.

        Returns:
            list[str]: A list of messages.
        """
        history = await self.get_history(user_id)
        return history.messages

    async def add_message(self, user_id: str, message: str) -> None:
        """
        Add a generic message to the chat history for a given user.

        Args:
            user_id (str): The unique identifier for the user.
            message (str): The message to add.
        """
        history = await self.get_history(user_id)
        history.add_message(message)

    async def add_ai_message(self, user_id: str, message: str) -> None:
        """
        Add an AI-generated message to the chat history for a given user.

        Args:
            user_id (str): The unique identifier for the user.
            message (str): The AI message to add.
        """
        history = await self.get_history(user_id)
        history.add_ai_message(message)

    async def add_user_message(self, user_id: str, message: str) -> None:
        """
        Add a user-generated message to the chat history for a given user.

        Args:
            user_id (str): The unique identifier for the user.
            message (str): The user message to add.
        """
        history = await self.get_history(user_id)
        history.add_user_message(message)

    async def clear_history(self, user_id: str) -> None:
        """
        Clear the chat history for a specific user.

        Args:
            user_id (str): The unique identifier for the user.
        """
        if user_id in store:
             del store[user_id]

    async def clear_full_history(self) -> None:
        """
        Clear the chat history for a specific user.

        Args:
            user_id (str): The unique identifier for the user.
        """
        global store
        store = {}

    async def get_all_histories(self) -> dict[str, BaseChatMessageHistory]:
        """
        Retrieve all histories.

        Returns:
            Dict[str, BaseChatMessageHistory]: A dictionary of all user histories.
        """
        return store


async def run_scheduler(history_manager: InMemoryHistory):
    """
    Start a scheduler to clear all chat histories periodically.
    """

    time = Config.HISTORY_CLEAR_SCHEDULE_IN_MINUTE

    schedule.every(time).minute.do(await history_manager.clear_full_history)

    def run_schedule():
        while True:
            schedule.run_pending()

    Thread(target=run_schedule, daemon=True).start()
