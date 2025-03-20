import os

from langchain_groq import ChatGroq
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonREPLTool
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from typing import Optional, Dict

os.environ["GROQ_API_KEY"] = "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"
# os.environ["DEEPSEEK_API_KEY"] = "sk-0b411f3148204371a17b68cd3e507103"
df = pd.read_csv("/content/sample_data/california_housing_train.csv")
llm = ChatGroq(model="llama3-70b-8192")
# pandas_agent = create_pandas_dataframe_agent(
#     llm,
#     df,
#     agent_type="tool-calling",
#     verbose=True,
#     allow_dangerous_code=True
# )