from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_
from src.database import models
from sqlalchemy.future import select
from fastapi import HTTPException as fastapi_HTTPException, status
from src.api.auth import utils
from src.api.chats.chats_schema import MessageSchema


async def create_user(user, db_: AsyncSession):
    user = user.dict()
    user["hashed_password"] = await utils.hash_password(user["hashed_password"])
    user_in_db = models.UserModel(**user)
    db_.add(user_in_db)
    await db_.commit()
    await db_.refresh(user_in_db)
    return user_in_db


async def validate_user(email: str, password: str, db_: AsyncSession) -> models.UserModel:
    query = select(models.UserModel).where(models.UserModel.email == email)
    result = await db_.execute(query)
    result = result.scalar()
    if not result:
        raise fastapi_HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=f"User with email: {email} doesn't exist")
    if not await utils.verify_password(password, result.hashed_password):
        raise fastapi_HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect password")
    else:
        return result


async def get_user_by_id(user_id: int, db_: AsyncSession) -> models.UserModel:
    query = select(models.UserModel).where(models.UserModel.id==user_id)
    result = await db_.execute(query)
    result = result.scalar()
    if not result:
        raise fastapi_HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='user not found')
    return result

async def get_users(db_: AsyncSession):
    query = select(models.UserModel).order_by(models.UserModel.id)
    result = await db_.execute(query)
    result = result.scalars().all()
    return result






