from typing import List

import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage
from langchain_experimental.agents import create_pandas_dataframe_agent
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")


class AgentState(TypedDict):
    query_type: str
    messages: list
    plots: list
    dataframe: pd.DataFrame
    answer: str


class EDAgent:
    def __init__(self):
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("query_analyzer", self.__query_analyzer)
        graph_builder.add_node("pandas_agent", self.__pandas_agent)
        graph_builder.add_node("plot_agent", self.__plot_agent)
        graph_builder.add_node("ml_agent", self.__ml_agent)

        graph_builder.add_edge(START, "query_analyzer")
        graph_builder.add_conditional_edges("query_analyzer",
                                            self.__query_choose,
                                            {"pandas_agent": "pandas_agent", "ml_agent": "ml_agent",
                                             "plot_agent": "plot_agent"})
        # graph_builder.add_edge("query_analyzer", "pandas_agent")
        graph_builder.add_edge("pandas_agent", END)
        graph_builder.add_edge("ml_agent", END)
        graph_builder.add_edge("plot_agent", END)
        self.plots_dir = "./plots"

        self.graph = graph_builder.compile()

    def invoke(self, messages: List[BaseMessage], dataframe: pd.DataFrame):
        return self.graph.invoke({"dataframe": dataframe, "messages": messages})

    def __query_analyzer(self, state: AgentState):
        print(state["messages"])
        last_message = state["messages"][-1]
        prompt_template = """
      У тебя есть следующий запрос от пользователя:\n
      {user_prompt}\n
      Определи, что нужно, чтобы ответить на заданный вопрос.\n
      Если можно ограничиться только pandas, в ответ напиши "pd"\n
      Если можно построить графики для удобства, напиши в ответ "plot"\n
      Если нужно применить инструменты машинного обучения, напиши "ml"\n
      В ответе должно быть только одно слово из это списка (pd, plot, ml) и больше ничего.
      """
        query_type = llm.invoke(prompt_template.format(user_prompt=last_message.content)).content
        print(query_type)
        return {'query_type': query_type}

    def __pandas_agent(self, state: AgentState):
        agent = create_pandas_dataframe_agent(
            llm,
            state["dataframe"],
            # agent_type="tool-calling",
            verbose=True,
            allow_dangerous_code=True
        )
        prompt_template = """
      У тебя следующий запрос от пользователя:\n
      {user_prompt}\n
      Дай ответ, предварительно проведя анализ df. 
      Если запрос слишком обширный, нет конкретики, напиши об этом в ответе. 
      Ответ верни на том же языке, на котором был запрос.
      """
        output = agent.invoke(prompt_template.format(user_prompt=state['messages'][-1].content))["output"]
        print(output)
        state["messages"].append(AIMessage(output))
        return {'answer': output}

    def __plot_agent(self, state: AgentState):
        user_prompt = state['messages'][-1].content
        code = llm.invoke("""Сгенерируй код для решения следующей задачи:\n
      {user_prompt}\n
      Все графики, которые возникают в твоем коде сохраняй в формате png в папке {plots_dir}. 
      В переменную plots сохрани пути до сохраненных изображений.
      Пиши код, считая, что переменная df уже инициализирована. В ответе напиши только исполняемый код и ничего более.
      """.format(user_prompt=user_prompt, plots_dir=self.plots_dir)).content
        code_clear = llm.invoke(
            "В данном сообщении оставь только код python:\n{code}".format(code=code)).content.replace("`", "").replace(
            "python", "")
        vars = {"df": state['dataframe']}
        print(code_clear)
        exec(code_clear, vars)
        plots = vars["plots"]
        return {"answer": "Графики предоставлены.", "plots": plots}

    def __ml_agent(self, state: AgentState):
        user_prompt = state['messages'][-1].content
        plots = []
        code = llm.invoke("""
      Ты специалист в области машинного обучения, ты мастерски владеешь библиотеками pandas, numpy, sklearn, matplotlib, seaborn. 
      Пользователь справшивает:\n
      {user_prompt}\n
      Подумай, какие методы машинного обучения будут наиболее релевантны в данной задаче. 
      Если будешь строить линейную модель, 
      выведи коэффициенты при переменных, если будешь строить дерево решений (ограничение глубины - 3), 
      выведи само дерево с помощью функции plot_tree() (обязательно используй эту функцию, 
      если будешь обучать решающее дерево, поставь параметры так, чтобы названия переменных на риснуке были подписаны).
      Выведи accuracy модели и сохрани его в переменную accuracy.
      Сохрани графики в формате .png в папку {plots_dir} 
      (проверяй, чтобы названия файлов были новыми, называй их просто числами по порядку).
      Добавь названия всех сохраненных изображений в лист plots.
      Не забудь, что не все признаки являются числовыми, для категориальных признаков используй one-hot-encoding.
      Считай, что переменная df у тебя уже инициализирована (сырой датафрейм)
      В ответе напиши только код python, который выполнит поставленную
      задачу.
      """.format(user_prompt=user_prompt, plots_dir=self.plots_dir))
        code_clear = llm.invoke(
            "В данном сообщении оставь только код python:\n{code}".format(code=code)).content.replace("`", "").replace(
            "python", "")
        vars = {"df": state['dataframe']}
        print(code_clear)
        exec(code_clear, vars)
        plots = vars["plots"]
        print(plots)

    def __query_choose(self, state: AgentState):
        if state["query_type"] == "pd":
            return "pandas_agent"
        if state["query_type"] == "plot":
            return "plot_agent"
        return "ml_agent"
