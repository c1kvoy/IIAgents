from datetime import datetime
from pydantic import BaseModel, ConfigDict


class MessageSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int | None = None
    chat_id: int
    user_id: int
    role: str
    message_text: str
    created_at: datetime
    image: str | None = None
class MessageOutSchema(BaseModel):
    role: str
    message_text: str
    image: str | None = None

class ChatSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int | None = None
    chat_id: int
    user_id: int
    csv_name: str

class ChatOutSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    chat_id: int
    csv_name: str