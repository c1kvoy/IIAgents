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
from src.api.analytics.analytics_schemas import (
    Message
)
from src.api.auth.auth_routers import authorize

analytics_router = APIRouter(dependencies=[Depends(authorize)])

UPLOAD_DIR = Path("../uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@analytics_router.post('/file')
async def post_file_router(file: UploadFile = File(...)) -> JSONResponse:
    if file.filename == '':
        raise FastAPIHTTPException(status_code=404, detail="No filename provided")
    if file and not file.filename.endswith('.csv'):
        raise FastAPIHTTPException(status_code=405, detail="File extension not supported")
    upload_path = UPLOAD_DIR / file.filename

    df = pd.read_csv(io.BytesIO(file.file.read()))
    print(df)
    df.to_csv(upload_path, index=False)
    response = agent_processing(df)
    return JSONResponse(content={"answer": response, "file_id": file.filename})


@analytics_router.post('/interact/{file_id}')
async def post_interact_router(context: list[Message], file_id: str) -> JSONResponse:
    if file_id == '':
        raise FastAPIHTTPException(status_code=404, detail="No filename provided")
    if file_id and not file_id.endswith('.csv'):
        raise FastAPIHTTPException(status_code=405, detail="File extension ion not supported")
    upload_path = UPLOAD_DIR / file_id
    df = pd.read_csv(upload_path)
    response = interact(context, df)
    return JSONResponse(content=response)


@analytics_router.post('/get_file')
async def post_file_router(file_name: str):
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