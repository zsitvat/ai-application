import json
import logging
import os
from datetime import datetime
from typing import Any

import redis.asyncio as redis
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

REDIS_URL_PREFIX = f"redis://{os.environ.get('RAG_REDIS_USER')}:{os.environ.get('RAG_REDIS_PASSWORD')}@{os.environ.get('RAG_REDIS_HOST')}:{os.environ.get('RAG_REDIS_PORT')}/"


class ChatHistoryService:
    async def get_chat_histories(
        self, chat_memory_db: str, chat_memory_index_name: str, user_id: str
    ):
        logging.getLogger("uvicorn").debug("Getting chat history for user: " + user_id)

        redis_connection_string = REDIS_URL_PREFIX + str(chat_memory_db)
        logging.getLogger("uvicorn").debug(
            "Connecting to redis: " + redis_connection_string
        )
        redis_client = redis.from_url(redis_connection_string)

        chat_histories = []

        try:
            cursor = 0
            while True:
                logging.getLogger("uvicorn").debug(
                    f"Scanning chat history for user: {user_id} with cursor: {cursor}"
                )
                cursor, keys = await redis_client.scan(
                    cursor,
                    match=f"{chat_memory_index_name}:{user_id}:*",
                    count=1000,
                )

                logging.getLogger("uvicorn").debug(f"Found keys: {keys}")
                for key in keys:
                    chat_history = await redis_client.lrange(key, 0, -1)

                    session_id = key.decode("utf-8").split(":")[-1]
                    title = json.loads(chat_history[0].decode("utf-8"))["data"][
                        "additional_kwargs"
                    ]["title"]
                    timestamp = json.loads(chat_history[0].decode("utf-8"))["data"][
                        "additional_kwargs"
                    ]["timestamp"]

                    chat_histories.append(
                        {
                            "user_id": user_id,
                            "session_id": session_id,
                            "title": title,
                            "timestamp": timestamp,
                        }
                    )

                if cursor == 0:
                    break
        finally:
            await redis_client.close()

        return sorted(
            chat_histories,
            key=lambda x: datetime.fromisoformat(x["timestamp"]),
            reverse=True,
        )

    async def create_chat_history(
        self,
        chat_memory_db: str,
        chat_memory_index_name: str,
        user_id: str,
        session_id: str,
        elements: list[Any],
    ):
        logging.getLogger("uvicorn").debug(
            f"Creating chat history for user: {user_id} and session: {session_id}"
        )

        redis_connection_string = REDIS_URL_PREFIX + str(chat_memory_db)
        logging.getLogger("uvicorn").debug(
            "Connecting to redis: " + redis_connection_string
        )
        redis_client = redis.from_url(redis_connection_string)

        try:
            for element in elements[::-1]:
                await redis_client.rpush(
                    f"{chat_memory_index_name}:{user_id}:{session_id}",
                    json.dumps(element),
                )

            logging.getLogger("uvicorn").debug(
                f"Created chat history for session: {session_id}"
            )
        finally:
            await redis_client.close()

        return elements

    async def get_chat_history(
        self,
        chat_memory_db: str,
        chat_memory_index_name: str,
        user_id: str,
        session_id: str,
    ):
        logging.getLogger("uvicorn").debug(
            f"Getting chat history for user: {user_id} and session: {session_id}"
        )

        redis_connection_string = REDIS_URL_PREFIX + str(chat_memory_db)
        logging.getLogger("uvicorn").debug(
            "Connecting to redis: " + redis_connection_string
        )
        redis_client = redis.from_url(redis_connection_string)

        try:
            chat_history = await redis_client.lrange(
                f"{chat_memory_index_name}:{user_id}:{session_id}", 0, -1
            )
            logging.getLogger("uvicorn").debug(
                f"Chat history for session: {session_id} is {chat_history}"
            )

            return [json.loads(entry.decode("utf-8")) for entry in chat_history[::-1]]
        finally:
            await redis_client.close()

    async def delete_chat_history(
        self,
        chat_memory_db: str,
        chat_memory_index_name: str,
        user_id: str,
        session_id: str | None,
    ):
        logging.getLogger("uvicorn").debug(
            f"Deleting chat history for user: {user_id} and session: {session_id}"
        )

        redis_connection_string = REDIS_URL_PREFIX + str(chat_memory_db)
        logging.getLogger("uvicorn").debug(
            "Connecting to redis: " + redis_connection_string
        )
        redis_client = redis.from_url(redis_connection_string)

        try:
            if session_id is None:
                async for key in redis_client.scan_iter(
                    f"{chat_memory_index_name}:{user_id}:*"
                ):
                    await redis_client.delete(key)
                logging.getLogger("uvicorn").debug(
                    f"Deleted all chat history for user: {user_id}"
                )
            else:
                await redis_client.delete(
                    f"{chat_memory_index_name}:{user_id}:{session_id}"
                )
                logging.getLogger("uvicorn").debug(
                    f"Deleted chat history for session: {session_id}"
                )
        finally:
            await redis_client.close()

        return True
