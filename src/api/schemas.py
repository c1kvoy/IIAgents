from pydantic import BaseModel

class Analytics(BaseModel):
    '''схема объекта аналитики которую мы будем переделывать'''
    average: int
    cor: float
    etc: str | int | float

class Message(BaseModel):
    role: str
    text: str