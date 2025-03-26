from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from src.api.auth.methods import *
from src.api.auth.auth_schemas import (
    UserInSchema,
    UserOutSchema,
)
from src.api.auth.utils import (
    create_access_token,
    create_refresh_token,
    jwt_decode,
    expire_validator,
    type_validator,
)

from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends
from src.core.config import settings
from src.database import methods as user_methods
from src.database import database
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="users/auth/login")


async def authorize(token: str = Depends(oauth2_scheme), db_=Depends(database.get_async_session)):
    payload = await jwt_decode(token=token)
    await expire_validator(payload=payload)
    await type_validator(payload['type'], settings.auth.ACCESS_TOKEN_TYPE)
    user = await get_user_by_id(int(payload['sub']), db_)
    return user.refresh_token


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
