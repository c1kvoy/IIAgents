from langchain_core.messages import BaseMessage


def clean_code(code: BaseMessage) -> str:
    return code.content.replace("```python", "").replace("```", "")
