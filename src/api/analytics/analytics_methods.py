from datetime import datetime
from pathlib import Path
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from fastapi import HTTPException as FastAPIHTTPException
from langchain_openai import ChatOpenAI
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
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llmimp = ChatOpenAI(model="o3-mini")

agent = EDAgent(llm)
agent_improved = EDAgent(llmimp)

async def agent_processing(user_id, chat_id, df, db) -> str:

    response = agent.invoke([SystemMessage(
        "Тебя зовут EDA_NA_DOM. Ты лучший аналитик данных и специалист в машинном обучении. Ответы присылай на русском языке!!!! Если вопрос слишком общий, то спроси какие-нибудь уточняющие детали."),
        HumanMessage("Сделай первичный анализ данных. Перечисли мне столбцы, их типы и количество наблюдений, предоставь общую информацию о представленных данных")], df)
    if "answer" in response:
        to_add = [
            MessageSchema(
                user_id=user_id,
                chat_id=chat_id,
                role="ai",
                message_text=response["answer"],
                created_at=datetime.now()
            )
        ]
        await add_message_from_chat(to_add, db)
    return response["answer"]


async def interact(model: str, user_id: int, chat_id: int, message: str, db: AsyncSession) -> dict[str, list | str]:
    # messages: list[MessageOutSchema] = await get_k_last_messages_from_chat(3, user_id, chat_id, db)
    # formatted_messages = [
    #     AIMessage(msg.message_text) if msg.role == "ai" else HumanMessage(msg.message_text)
    #     for msg in messages
    # ]

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

    if model == "default":
        print("4o model")
        response = agent.invoke([SystemMessage(
            "Тебя зовут EDA_NA_DOM. Ты лучший аналитик данных и специалист в машинном обучении. Ответы присылай на русском языке!!!"
        ), HumanMessage(message)] , df)
    else:
        print("o3 model")
        response = agent_improved.invoke([SystemMessage(
            "Тебя зовут EDA_NA_DOM. Ты лучший аналитик данных и специалист в машинном обучении. Ответы присылай на русском языке!!!"
        ), HumanMessage(message)], df)

    if response["answer"]:
        if "plots" in response and response["plots"][0]:
            ai_msg = MessageSchema(user_id=user_id, chat_id=chat_id, role="ai", message_text=response["answer"], created_at=datetime.now(), image=response['plots'][0])
        else:
            ai_msg = MessageSchema(user_id=user_id, chat_id=chat_id, role="ai", message_text=response["answer"],
                                   created_at=datetime.now())
        to_add.append(ai_msg)

    await add_message_from_chat(to_add, db)
    print(response["answer"])
    final_response = {
        "plots": response.get("plots", []),
        "answer": response.get("answer", "Ответ не найден")
    }
    return final_response