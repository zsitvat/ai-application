import json
import logging
import os
import re

import httpx
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage


class DataChatHistoryService:

    async def get_history_messages_as_string(
        self, max_history_lenght=20, uuid: str = None, name_tags=True, reverse=False
    ):
        messages = await self.get_messages_from_chat_history(uuid)

        history_messages = ""
        history_lenght = len(messages)
        start_at = history_lenght - max_history_lenght - 1

        if start_at < 0:
            start_at = 0
        index = 0

        for message in messages[::-1]:
            if index < history_lenght and index >= start_at:
                if name_tags:
                    if reverse:
                        history_messages = (
                            message["type"].upper()
                            + ": "
                            + message["messageAsText"]
                            + "\n\n"
                            + history_messages
                        )
                    else:
                        history_messages += (
                            message["type"].upper()
                            + ": "
                            + message["messageAsText"]
                            + "\n\n"
                        )
                else:
                    if reverse:
                        history_messages = (
                            f"{message['messageAsText']} {history_messages}"
                        )
                    else:
                        history_messages += f"{message['messageAsText']} "

            elif index >= history_lenght:
                break
            index += 1

        return history_messages

    async def get_human_messages_as_string(
        self, max_history_lenght=20, uuid: str = None, reverse=False
    ):
        messages = await self.get_messages_from_chat_history(uuid)

        history_messages = ""
        history_lenght = len(messages)
        start_at = history_lenght - max_history_lenght - 1

        if start_at < 0:
            start_at = 0
        index = 0

        for message in messages[::-1]:
            if index < history_lenght and index >= start_at:
                if message["type"] == "human":
                    if reverse:
                        history_messages = (
                            f"{message['messageAsText']} | {history_messages}"
                        )
                    else:
                        history_messages += f"{message['messageAsText']} | "

            elif index >= history_lenght:
                break
            index += 1

        return history_messages

    async def get_ai_messages_as_string(self, max_history_lenght=20, uuid: str = None):
        messages = await self.get_messages_from_chat_history(uuid)

        history_messages = ""
        history_lenght = len(messages)
        start_at = history_lenght - max_history_lenght - 1

        if start_at < 0:
            start_at = 0
        index = 0

        for message in messages:
            if index < history_lenght and index >= start_at:
                if message["type"] == "ai":
                    history_messages = (
                        message["messageAsText"] + " | " + history_messages
                    )
            elif index >= history_lenght:
                break
            index += 1

        return history_messages

    async def get_history_messages_as_dict(self, uuid: str, reverse=False):
        if reverse:
            messages = await self.get_messages_from_chat_history(uuid)
            messages.reverse()
        else:
            messages = await self.get_messages_from_chat_history(uuid)
        return messages

    async def get_history_messages_as_list(self, uuid: str, reverse=False):

        if reverse:
            messages = await self.get_messages_from_chat_history(uuid)
            messages.reverse()
        else:
            messages = await self.get_messages_from_chat_history(uuid)

        messages_as_list = []
        for message in messages:
            if message["type"] == "ai":
                message_as_class = AIMessage(message["messageAsText"])
            else:
                message_as_class = HumanMessage(message["messageAsText"])
            messages_as_list.append(message_as_class)
        return messages_as_list

    async def get_ai_messages_as_list(self, uuid: str, reverse=False):

        if reverse:
            messages = await self.get_messages_from_chat_history(uuid)
            messages.reverse()
        else:
            messages = await self.get_messages_from_chat_history(uuid)

        messages_as_list = []
        for message in messages:
            if message["type"] == "ai":
                messages_as_list.append(AIMessage(message["messageAsText"]))

        return messages_as_list

    async def get_human_messages_as_list(self, uuid: str, reverse=False):

        if reverse:
            messages = await self.get_messages_from_chat_history(uuid)
            messages.reverse()
        else:
            messages = await self.get_messages_from_chat_history(uuid)

        messages_as_list = []
        for message in messages:
            if message["type"] == "human":
                messages_as_list.append(HumanMessage(message["messageAsText"]))

        return messages_as_list

    async def get_ai_messages(self, uuid, reverse=False):
        if reverse:
            messages = await self.get_messages_from_chat_history(uuid)
            messages.reverse()
        else:
            messages = await self.get_messages_from_chat_history(uuid)
        history_messages = []
        for message in messages:
            if message["type"] == "ai":
                history_messages.append(message["messageAsText"])
        return history_messages

    async def get_human_messages(self, uuid: str, reverse=False):
        if reverse:
            messages = await self.get_messages_from_chat_history(uuid)
            messages.reverse()
        else:
            messages = await self.get_messages_from_chat_history(uuid)
        history_messages = []
        for message in messages:
            if message["type"] == "human":
                history_messages.append(message["messageAsText"])
        return history_messages

    def save_inputs_to_db(self, question, answer, session_id=None):
        """Save chat inputs to llm history d[A-Z]"""

        host = os.environ.get("HISTORY_DB_HOST")
        dbname = os.environ.get("HISTORY_DB_NAME")
        port = os.environ.get("HISTORY_DB_PORT")
        user = os.environ.get("HISTORY_DB_USER")
        pwd = os.environ.get("HISTORY_DB_PASSWORD")

        connection_string = "postgres://{user}:{pwd}@{host}:{port}/{dbname}".format(
            user=user, pwd=pwd, host=host, port=port, dbname=dbname
        )

        history = PostgresChatMessageHistory(
            connection_string=connection_string, session_id=session_id
        )
        logging.getLogger("uvicorn").info("Creates llm history db connection!")

        history.add_user_message(question)
        history.add_ai_message(answer)

    async def get_conversations(self, uuid: str):
        """Get conversations from chat history d[A-Z]"""

        limit = 100
        page = 0
        size = 100

        base = os.environ.get("DATA_API_BASE_URL")
        path = os.environ.get("CHAT_HISTORY_REST_API_ROUTE_PATH").format(
            uuid=uuid, limit=limit, page=page, size=size
        )

        url = base + path

        headers = {
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=30.0)

            if response.status_code == 200:
                return response
            else:
                raise httpx.HTTPStatusError(
                    f"Request failed with status {response.status_code}",
                    request=response.request,
                    response=response,
                )

    async def get_messages_from_chat_history(self, uuid: str):
        """Get messages from chat history db and return as list of dict[A-Z]"""

        response = await self.get_conversations(uuid)
        if response is None:
            raise ValueError("Error while getting messages from chat history db!")
        response_data = response.json()

        messages = []
        ignore_messages = os.environ.get("IGNORED_MESSAGES")

        ignore_messages = (
            json.loads(ignore_messages) if ignore_messages is not None else list("hi")
        )

        for object in response_data["content"]:
            if object["isOutgoingMessage"]:
                answer_type = "ai"
            else:
                answer_type = "human"

            message_as_text = ""
            if object["messageAsRaw"] == "[SendTextWithButtons]":
                message_as_text = json.loads(object["messageAsText"])["text"]
            elif object["messageAsRaw"] == "[TextWithQuickReplies]":
                message_as_text = json.loads(object["messageAsText"])["text"]
            elif (
                "sendTextWithButtons" in object["messageAsText"]
                or "sendHangoutsChatButtonTemplate" in object["messageAsText"]
            ):
                message_as_text = json.loads(object["messageAsText"])["text"]
            elif (
                "genericTemplate" in object["messageAsText"]
                or "genericTemplate" in object["messageAsRaw"]
            ):
                continue
            elif re.match("##INIT.*", object["messageAsText"]) or re.match(
                "##INIT.*", object["messageAsRaw"]
            ):
                continue
            elif "Image:" in object["messageAsText"]:
                continue
            elif "Error:" in object["messageAsRaw"]:
                continue
            elif "Exception type:" in object["messageAsRaw"]:
                continue
            elif str(object["messageAsText"]) in ignore_messages:
                continue
            elif object["messageAsRaw"] == "[Livechat generic template]":
                continue
            elif "DEBUG" in object["messageAsText"]:
                lines = object["messageAsText"].splitlines()
                message_as_text = lines[3]
            else:
                message_as_text = object["messageAsText"]

            if message_as_text != "" or message_as_text is not None:
                messages.append(
                    {
                        "type": answer_type,
                        "messageAsText": message_as_text,
                        "messageAsRaw": object["messageAsRaw"],
                    }
                )

        return messages
