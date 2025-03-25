from langchain_core.messages import SystemMessage, HumanMessage
from fastapi import HTTPException as FastAPIHTTPException
from pandas.core.interchange.dataframe_protocol import DataFrame

from src.api.analytics.analytics_schemas import (
    Message,
)
from src.llm.agents import (
    EDAgent
)
agent = EDAgent()
def agent_processing(df) -> str:
    '''возвращает статус процессинга (нужно или нет переспрашивать пользователя) и вопрос, интересуующий ллмку'''
    response = agent.invoke([SystemMessage(
        "Тебя зовут EDA_NA_DOM. Ты лучший аналитик данных и специалист в машинном обучении. Ответы присылай на русском языке!!!! Если вопрос слишком общий, то спроси какие-нибудь уточняющие детали."),
        HumanMessage("Сделай первичный анализ данных. Перечисли мне столбцы, их типы и количество наблюдений, предоставь общую информацию о представленных данных")], df)
    return response["answer"]


def interact(context: list[Message], df: DataFrame) -> dict[str, list | str]:
    latest_mes = context[-1] if context[-1].role == 'human' else context[-2]
    # Wrap the message into a state dictionary for validation:
    is_cool = agent.validate_prompt({"messages": [HumanMessage(latest_mes.text)]})
    if not is_cool:
        raise FastAPIHTTPException(status_code=404, detail="unhealthy behavior")
    response = agent.invoke([SystemMessage(
        "Тебя зовут EDA_NA_DOM. Ты лучший аналитик данных и специалист в машинном обучении. Ответы присылай на русском языке!!!"),
        HumanMessage(latest_mes.text)],
        df
    )
    print(response["answer"])
    final_response = {
        "plots": response.get("plots", []),
        "answer": response.get("answer", "Ответ не найден")
    }
    return final_response