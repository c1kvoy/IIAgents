from pathlib import Path
from fastapi import (
    APIRouter,
    HTTPException as FastAPIHTTPException,
    UploadFile,
    File,
    Depends,
    status
)
import io

from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
from src.api.analytics.analytics_methods import (
    agent_processing,
    interact
)
from src.api.analytics.analytics_schemas import Message
from src.api.chats.chats_methods import add_chat, get_csv_by_id
from src.database.database import get_async_session
from src.api.auth.auth_routers import authorize

analytics_router = APIRouter()

UPLOAD_DIR = Path("../uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@analytics_router.post('/file')
async def post_file_router(user_id: int, file: UploadFile = File(...), db_ = Depends(get_async_session), validate_id: int = Depends(authorize)) -> JSONResponse:
    if user_id != validate_id:
        raise FastAPIHTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user_id")
    if file.filename == '':
        raise FastAPIHTTPException(status_code=404, detail="No filename provided")
    if file and not file.filename.endswith('.csv'):
        raise FastAPIHTTPException(status_code=405, detail="File extension not supported")
    upload_path = UPLOAD_DIR / file.filename

    df = pd.read_csv(io.BytesIO(file.file.read()))
    print(df)
    df.to_csv(upload_path, index=False)

    chat_id = await add_chat(user_id, file.filename, db_)
    response = await agent_processing(user_id, chat_id, df, db_)
    return JSONResponse(content={"answer": response, "file_name": file.filename,"chat_id": chat_id})


@analytics_router.post('/interact/')
async def post_interact_router(user_id_: int, chat_id: int, message_: Message, model: str, db_ = Depends(get_async_session), validate_id: int = Depends(authorize)) -> JSONResponse:
    if user_id_ != validate_id:
        raise FastAPIHTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user_id")
    response = await interact(model,user_id_, chat_id, message_.text, db_)
    return JSONResponse(content=response)


@analytics_router.post('/get_file')
def post_file_router(file_name: str, validate_id: int = Depends(authorize)):
    file_name = Path(file_name).name

    if file_name == '':
        raise FastAPIHTTPException(status_code=404, detail="No filename provided")
    supported_extensions = {
        '.csv': 'text/csv',
        '.png': 'image/png'
    }
    file_extension = Path(file_name).suffix.lower()

    if file_extension not in supported_extensions:
        raise FastAPIHTTPException(status_code=405, detail="File extension not supported")

    upload_path = UPLOAD_DIR / file_name
    if not upload_path.exists():
        raise FastAPIHTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=upload_path,
        filename=file_name,
        media_type=supported_extensions[file_extension]
    )
