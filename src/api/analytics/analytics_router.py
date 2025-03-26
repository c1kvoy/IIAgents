from pathlib import Path
from fastapi import (
    APIRouter,
    HTTPException as FastAPIHTTPException,
    UploadFile,
    File,
    Depends,
)
import io

from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
from src.api.analytics.analytics_methods import (
    agent_processing,
    interact
)
from src.api.chats.chats_methods import add_chat, get_csv_by_id
from src.database.database import get_async_session
from src.api.auth.auth_routers import authorize

analytics_router = APIRouter(dependencies=[Depends(authorize)])

UPLOAD_DIR = Path("../uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@analytics_router.post('/file')
async def post_file_router(user_id: int, chat_id: int, file: UploadFile = File(...), db_ = Depends(get_async_session)) -> JSONResponse:
    if file.filename == '':
        raise FastAPIHTTPException(status_code=404, detail="No filename provided")
    if file and not file.filename.endswith('.csv'):
        raise FastAPIHTTPException(status_code=405, detail="File extension not supported")
    upload_path = UPLOAD_DIR / file.filename

    df = pd.read_csv(io.BytesIO(file.file.read()))
    print(df)
    df.to_csv(upload_path, index=False)
    response = agent_processing(df)
    await add_chat(user_id, chat_id, file.filename, db_)
    return JSONResponse(content={"answer": response, "file_id": file.filename})


@analytics_router.post('/interact/')
async def post_interact_router(user_id_: int, chat_id: int, message_: str, db_ = Depends(get_async_session)) -> JSONResponse:
    response = await interact(user_id_, chat_id, message_, db_)
    return JSONResponse(content=response)


@analytics_router.post('/get_file/')
async def post_file_router(user_id, chat_id, db_ = Depends(get_async_session)) -> FileResponse:
    file_name = await get_csv_by_id(user_id, chat_id, db_)
    if file_name == '':
        raise FastAPIHTTPException(status_code=404, detail="No filename provided")

    supported_extensions = {
        '.csv': 'text/csv',
        '.png': 'image/png'
    }

    file_extension = Path(file_name).suffix.lower()

    upload_path = UPLOAD_DIR / file_name
    if not upload_path.exists():
        raise FastAPIHTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=upload_path,
        filename=file_name,
        media_type=supported_extensions[file_extension]
    )