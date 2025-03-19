
async def agent_processing() -> [bool, str]:
    '''возвращает статус процессинга (нужно или нет переспрашивать пользователя) и вопрос, интересуующий ллмку'''
    return True, "question"

async def agent_validate() -> [bool, str]:
    '''возвращает статус(валиден ли запрос) и как нужно улучшить запрос'''
    return [True,'prompt']
