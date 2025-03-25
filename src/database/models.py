from datetime import datetime

from sqlalchemy import ForeignKey

from src.database.database import Base
from sqlalchemy.orm import mapped_column, Mapped, relationship

class UserModel(Base):
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(primary_key=True, index=True, nullable=False, autoincrement=True)
    email: Mapped[str] = mapped_column(index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(nullable=False)
    refresh_token: Mapped[str] = mapped_column(nullable=True)

    posts: Mapped[list["MessageModel"]] = relationship("MessageModel", back_populates="user")

class MessageModel(Base):
    __tablename__ = 'messages'

    id: Mapped[int] = mapped_column(primary_key=True, index=True, nullable=False)
    chat_id: Mapped[int] = mapped_column(index=True, nullable=False)
    message_text: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    user: Mapped["UserModel"] = relationship("UserModel", back_populates="posts")
