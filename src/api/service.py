from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.analytics.analytics_router import analytics_router
from src.api.auth.auth_routers import auth_router
from src.api.chats.chats_routers import chats_router
from src.database.database import create_tables

app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analytics_router)
app.include_router(auth_router)
app.include_router(chats_router)

@app.on_event("startup")
async def on_startup():
    await create_tables()