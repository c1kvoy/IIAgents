from datetime import datetime
from pydantic import BaseModel, ConfigDict


class MessageSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    chat_id: int
    message_text: str
    created_at: datetime

