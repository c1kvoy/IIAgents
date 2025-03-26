from src.api.chats.chats_schema import MessageSchema, ChatSchema
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from src.database import models
from sqlalchemy import and_
from src.database.models import MessageModel
from fastapi.exceptions import HTTPException as FastAPIHTTPException


async def add_message_from_chat(messages: list[MessageSchema], db_: AsyncSession):
    to_db = [MessageModel(**msg.model_dump()) for msg in messages]
    for msg in to_db:
        db_.add(msg)

    await db_.commit()

    for msg in to_db:
        await  db_.refresh(msg)
    return

async def add_chat(user_id: int, chat_id: int, file_name: str, db_: AsyncSession):
    print(chat_id, file_name)
    data = models.ChatModel(**ChatSchema(user_id=user_id, chat_id=chat_id, csv_name=file_name).model_dump())
    db_.add(data)
    await db_.commit()
    await db_.refresh(data)
    return

async def get_csv_by_id(user_id: int, chat_id: int, db_: AsyncSession) -> str:
    q = select(models.ChatModel).where(and_(models.ChatModel.user_id == user_id, models.ChatModel.chat_id == chat_id))
    data = await db_.execute(q)
    data = data.scalars().first()
    if data is None:
        raise FastAPIHTTPException(status_code=404, detail="Chat not found")
    return data.csv_name

async def get_k_last_messages_from_chat(k_: int, user_id_: int, chat_id_: int, db_: AsyncSession) -> list[
    MessageSchema]:
    print(k_, user_id_, chat_id_)
    query = (
        select(models.MessageModel)
        .where(
            models.MessageModel.user_id == user_id_,
            models.MessageModel.chat_id == chat_id_
        )
        .order_by(models.MessageModel.created_at.asc())  # Берем старые сначала
        .limit(k_)
    )

    data = await db_.execute(query)
    messages = data.scalars().all()
    print(messages)

    return [MessageSchema.model_validate(msg) for msg in messages]

async def get_all_chat(db_: AsyncSession) -> list[MessageSchema]:
    q = select(models.MessageModel).order_by(models.MessageModel.chat_id)
    data = await db_.execute(q)
    data = data.scalars().all()
    print(data)
    return [MessageSchema.model_validate(msg) for msg in data]