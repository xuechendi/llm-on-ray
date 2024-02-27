from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

model_name = "mistral-7b-instruct-v0.2"
tools = [DuckDuckGoSearchRun(max_results=1)]


prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
model = ChatOpenAI(
    openai_api_base="http://localhost:8000/v1",
    model_name=model_name,
    openai_api_key="not_needed",
    temperature=0,
).bind_tool(tools)

runnable = prompt | model

runnable.invoke({"input": "what is the weather in sf"})
