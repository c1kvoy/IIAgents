from typing import List
from src.api.schemas import (
    Analytics
)

async def agent_processing(id: str) -> [bool, str]:
    '''возвращает статус процессинга (нужно или нет переспрашивать пользователя) и вопрос, интересуующий ллмку'''
    return True, "question"

async def agent_validate(prompt: str) -> [bool, str]:
    '''возвращает статус(валиден ли запрос) и как нужно улучшить запрос + проверка безопасности'''
    return [True,'prompt']


async def agent_analysis() -> dict[str, str | Analytics]:
    '''логика агента подведения итогов парсим объект аналитики, возвращаем численную аналитику и выводы агента'''
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
