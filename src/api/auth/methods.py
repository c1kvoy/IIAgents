from fastapi.security import OAuth2PasswordBearer

from src.api.auth.utils import (
    jwt_decode,
    expire_validator,
    type_validator,
)
from fastapi import Depends
from src.core.config import settings
from src.database import methods as user_methods
from src.database import database
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="users/auth/login")


async def authorize(token: str = Depends(oauth2_scheme), db_=Depends(database.get_async_session)):
    payload = await jwt_decode(token=token)
    await expire_validator(payload=payload)
    await type_validator(payload['type'], settings.auth.ACCESS_TOKEN_TYPE)
    user = await user_methods.get_user_by_id(int(payload['sub']), db_)
    return user.refresh_token