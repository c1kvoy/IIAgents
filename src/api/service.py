from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers.analytics import analytics_router
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

@app.on_event("startup")
async def on_startup():
    await create_tables()