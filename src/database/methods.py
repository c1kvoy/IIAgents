from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_
from src.database.models import (
    MessageModel
)
from sqlalchemy.future import select
from src.api.schemas import(
    chats
)

async def get_k_last_messages_from_chat(k_: int, user_id_: int, chat_id_: int, db_: AsyncSession) -> list[chats.MessageSchema]:
    query = select(MessageModel)\
        .where(and_(MessageModel.user_id == user_id_, MessageModel.chat_id == chat_id_))\
        .order_by(MessageModel.created_at.desc())\
        .limit(k_)

    data = await db_.execute(query)
    messages = data.scalars().all()

    return [chats.MessageSchema.model_validate(msg) for msg in messages]





