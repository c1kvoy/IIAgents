from pydantic import BaseModel


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

    class Config:
        from_attributes = True

class TokenTypeSchema(BaseModel):
    token: str
    token_type: str