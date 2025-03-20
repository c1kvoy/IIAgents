
from langchain_core.messages import SystemMessage, HumanMessage
from fastapi import HTTPException as FastAPIHTTPException
from pandas.core.interchange.dataframe_protocol import DataFrame

from src.api.schemas import (
    Analytics,
    Message,
)
from src.llm.agents import (
    EDAgent
)
agent = EDAgent()
async def agent_processing(df) -> str:
    '''возвращает статус процессинга (нужно или нет переспрашивать пользователя) и вопрос, интересуующий ллмку'''
    response = agent.invoke([SystemMessage(
        "Тебя зовут EDA_NA_DOM. Ты лучший аналитик данных и специалист в машинном обучении. Ответы присылай на русском языке!!!! Если вопрос слишком общий, то спроси какие-нибудь уточняющие детали."),
        HumanMessage("Сделай первичный анализ данных. Перечисли мне столбцы, их типы и количество наблюдений, предоставь общую информацию о представленных данных")], df)
    return response["answer"]


async def interact(context: list[Message], df: DataFrame) -> dict[str, list | str]:
    latest_mes = context[-1] if context[-1].role == 'human' else context[-2]
    is_cool = agent.validate_prompt(latest_mes.text)
    if not is_cool:
        raise FastAPIHTTPException(status_code=404, detail="unhealthy behavior")
    response = agent.invoke([SystemMessage(
        "Тебя зовут EDA_NA_DOM. Ты лучший аналитик данных и специалист в машинном обучении. Ответы присылай на русском языке!!!"),
        HumanMessage(latest_mes.text)],
        df
    )
    final_response: dict[str, list|str] = {"plots": [], "answer": response["answer"]}
    if response["plots"]:
        final_response["plots"] = response["plots"]
    return final_response

# async def agent_validate(prompt: str) -> [bool, str]:
#     '''возвращает статус(валиден ли запрос) и как нужно улучшить запрос + проверка безопасности'''
#     return [True,'prompt']
#
#
# async def agent_analysis() -> dict[str, str | Analytics]:
#     '''логика агента подведения итогов парсим объект аналитики, возвращаем численную аналитику и выводы агента'''
#     return {
#         "analysis":  {
#             "average": 3212,
#             "cor": .1,
#             "etc": "etc"
#         },
#         "conclusions": "conclusions"
#     }
#
# async def agent_visualise(c: dict) -> dict:
#     '''логика визаулизации и тд'''
#     return {
#         "data": "hueta"
#     }
