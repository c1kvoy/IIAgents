from fastapi import APIRouter, Depends, status
from src.database.database import get_async_session
from src.api.chats.chats_schema import MessageSchema, MessageOutSchema
from src.api.chats.chats_methods import (
    get_k_last_messages_from_chat, get_all_chat
)

chats_router = APIRouter(prefix="/chats", tags=["chats"])



@chats_router.post('/last_messages')
async def get_last_messages(k_: int, chat_id_: int, user_id_: int, db_ = Depends(get_async_session)) -> list[MessageOutSchema]:
    messages = await get_k_last_messages_from_chat(k_, user_id_, chat_id_, db_)
    return messages
