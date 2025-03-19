from typing import List
from src.api.schemas import (
    Analytics
)

async def agent_processing() -> [bool, str]:
    '''возвращает статус процессинга (нужно или нет переспрашивать пользователя) и вопрос, интересуующий ллмку'''
    return True, "question"

async def agent_validate() -> [bool, str]:
    '''возвращает статус(валиден ли запрос) и как нужно улучшить запрос + проверка безопасности'''
    return [True,'prompt']

async def agent_analysis() -> Analytics:
    '''логика агента анализа'''
    return Analytics(
        average=3212,
        cor=0.1,
        etc="etc"
    )

async def agent_conclusions(c: Analytics) -> dict[str, str | Analytics]:
    '''логика агента подведения итогов парсим объект аналитики'''
    return {
        "analysis":  {
            "average": 3212,
            "cor": .1,
            "etc": "etc"
        },
        "conclusions": "conclusions"
    }

async def agent_visualise(c: dict) -> dict:
    '''логика визаулизации и тд'''
    return {
        "data": "hueta"
    }
