from pydantic import BaseModel

class Analytics(BaseModel):
    '''схема объекта аналитики которую мы будем переделывать'''
    average: int
    cor: float
    etc: str | int | float

class Message(BaseModel):
    role: str
    text: str

class UserDBSchema(BaseModel):
    id: int
    email: str
    hashed_password: str
    refresh_token: str

class UserInSchema(BaseModel):
    email: str
    hashed_password: str

class UserOutSchema(BaseModel):
    id: int
    email: str


class TokenTypeSchema(BaseModel):
    token: str
    token_type: str