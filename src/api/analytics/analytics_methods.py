from datetime import datetime
from pathlib import Path
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from fastapi import HTTPException as FastAPIHTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.chats.chats_schema import MessageSchema, MessageOutSchema
from src.llm.agents import (
    EDAgent
)

from src.api.chats.chats_methods import (
    get_k_last_messages_from_chat,
    add_message_from_chat, get_csv_by_id
)


UPLOAD_DIR = Path("../uploads")
agent = EDAgent()


def agent_processing(df) -> str:

    response = agent.invoke([SystemMessage(
        "Тебя зовут EDA_NA_DOM. Ты лучший аналитик данных и специалист в машинном обучении. Ответы присылай на русском языке!!!! Если вопрос слишком общий, то спроси какие-нибудь уточняющие детали."),
        HumanMessage("Сделай первичный анализ данных. Перечисли мне столбцы, их типы и количество наблюдений, предоставь общую информацию о представленных данных")], df)
    return response["answer"]


async def interact(user_id: int, chat_id: int, message: str, db: AsyncSession) -> dict[str, list | str]:
    messages: list[MessageOutSchema] = await get_k_last_messages_from_chat(3, user_id, chat_id, db)
    formatted_messages = [
        AIMessage(msg.message_text) if msg.role == "ai" else HumanMessage(msg.message_text)
        for msg in messages
    ]

    is_cool = agent.validate_prompt(message)
    if not is_cool:
        raise FastAPIHTTPException(status_code=404, detail="unhealthy behavior")

    #better_message = agent.get_prompt_better(message)

    file_name = await get_csv_by_id(user_id, chat_id, db)
    upload_path = UPLOAD_DIR / file_name
    df = pd.read_csv(upload_path)

    to_add = [
        MessageSchema(
            user_id=user_id,
            chat_id=chat_id,
            role="user",
            message_text=message,
            created_at=datetime.now()
        )
    ]

    formatted_messages.append(HumanMessage(message))

    response = agent.invoke([SystemMessage(
        "Тебя зовут EDA_NA_DOM. Ты лучший аналитик данных и специалист в машинном обучении. Ответы присылай на русском языке!!!"
    )] + formatted_messages, df)

    if response["answer"]:
        ai_msg = MessageSchema(user_id=user_id, chat_id=chat_id, role="ai", message_text=response["answer"], created_at=datetime.now())
        to_add.append(ai_msg)

    await add_message_from_chat(to_add, db)

    print(response["answer"])
    final_response = {
        "plots": response.get("plots", []),
        "answer": response.get("answer", "Ответ не найден")
    }
    return final_response