import io
from fastapi import FastAPI, File, UploadFile, HTTPException as FastAPIHTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import pandas as pd
from src.api.schemas import (
    Analytics
)
from src.api.methods import (
    agent_processing,
    agent_validate,
    agent_analysis,
    agent_visualise,
)

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


@app.get('/agent_processing/{file_id}')
async def get_agent_processing(file_id: str) -> dict[str, bool | str]:
    should_continue, question = await agent_processing(file_id)
    return { "status": should_continue, "question": question }


@app.post('/agent_validate')
async def get_agent_validate(prompt: str) -> dict[str, bool | str]:
    status, issue = await agent_validate(prompt)
    return { "status": status, "issues": issue}


@app.get('/agent_analysis')
async def get_agent_analysis() -> dict[str, Analytics | str]:
    '''сухие численные данные по типу корреляция и тд'''
    results = await agent_analysis()
    return results

@app.post('/agent_visualise')
async def get_agent_visualise(context: dict) -> dict[str, bool]:
    '''выбросы аномалии и тд'''
    results = await agent_visualise(context)
    return JSONResponse(content=results)