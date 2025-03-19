import csv
import io

from src.api.methods import (
    agent_processing,
    agent_validate
)
from fastapi import FastAPI, File, UploadFile, HTTPException as FastAPIHTTPException
from pathlib import Path
import pandas as pd
app = FastAPI()

UPLOAD_DIR = Path("../uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post('/file')
async def post_file_router(file: UploadFile = File(...)) -> dict[str, int | Path]:
    if file.filename == '':
        raise FastAPIHTTPException(status_code=404, detail="No filename provided")
    if file and not file.filename.endswith('.csv'):
        raise FastAPIHTTPException(status_code=405, detail="File extension not supported")
    upload_path = UPLOAD_DIR / file.filename

    df = pd.read_csv(io.BytesIO(file.file.read()))
    print(df)
    df.to_csv(upload_path, index=False)
    return {"status": 200, "filename": upload_path}


@app.get('/agent_processing')
async def get_agent_processing() -> dict[str, bool]:
    should_continue, question = await agent_processing()
    return { "status": should_continue, "question": question }


@app.get('/agent_validate')
async def get_agent_validate(prompt: str) -> dict[str, bool | str]:
    status, issue = await agent_validate()
    return { "status": status, "issues": issue}

