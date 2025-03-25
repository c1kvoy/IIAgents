from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm

from src.database import database
from src.database.methods import *
from src.api.auth.auth_schemas import (
    UserInSchema,
    UserOutSchema,
)
from src.api.auth.utils import (
    create_access_token,
    create_refresh_token,
)

auth_router = APIRouter(tags=['auth'], prefix='/users/auth')


@auth_router.post('/register', response_model=UserOutSchema)
async def create_user_router(user: UserInSchema, db_=Depends(database.get_async_session)) -> UserOutSchema:
    user_ = await create_user(user, db_)
    return UserOutSchema.from_orm(user_)


@auth_router.post('/login')
async def login_router(form: OAuth2PasswordRequestForm = Depends(), db_=Depends(database.get_async_session)):
    user_from_db = await validate_user(form.username, form.password, db_)
    payload = {
        'sub': str(user_from_db.id),
        'email': user_from_db.email,
    }
    user_from_db.refresh_token = await create_refresh_token(payload)
    access_token = await create_access_token(payload)
    db_.add(user_from_db)
    await db_.commit()
    return {
        'access_token': access_token,
        'token_type': 'bearer',
    }
