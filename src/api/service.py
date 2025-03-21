import io
from fastapi import FastAPI, File, UploadFile, HTTPException as FastAPIHTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import (
    Analytics,
    Message,
)
from src.api.methods import (
    agent_processing,
    interact,

)

app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("../uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post('/file')
def post_file_router(file: UploadFile = File(...)) -> JSONResponse:
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


@app.post('/interact/{file_id}')
def post_interact_router(context: list[Message], file_id: str) -> JSONResponse:
    if file_id == '':
        raise FastAPIHTTPException(status_code=404, detail="No filename provided")
    if file_id and not file_id.endswith('.csv'):
        raise FastAPIHTTPException(status_code=405, detail="File extension ion not supported")
    upload_path = UPLOAD_DIR / file_id
    df = pd.read_csv(upload_path)
    response = interact(context, df)
    return JSONResponse(content=response)

from fastapi.responses import FileResponse

@app.post('/get_file')
def post_file_router(file_name: str):
    if file_name == '':
        raise FastAPIHTTPException(status_code=404, detail="No filename provided")
    if file_name and not file_name.endswith('.csv'):
        raise FastAPIHTTPException(status_code=405, detail="File extension not supported")
    upload_path = UPLOAD_DIR / file_name
    if not upload_path.exists():
        raise FastAPIHTTPException(status_code=404, detail="File not found")
    return FileResponse(path=upload_path, filename=file_name, media_type='text/csv')