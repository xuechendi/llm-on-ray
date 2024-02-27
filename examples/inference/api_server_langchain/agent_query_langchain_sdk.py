from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain import hub

model_name = "mistral-7b-instruct-v0.2"
tools = [DuckDuckGoSearchRun(max_results=1)]
prompt = hub.pull("hwchase17/openai-tools-agent")
print(prompt)
llm = ChatOpenAI(
    openai_api_base="http://localhost:8000/v1",
    model_name=model_name,
    openai_api_key="not_needed",
    max_tokens=512,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

###############################     Old API    #################################
# agent = initialize_agent(
#         tools=tools,
#         llm=llm,
#         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         handle_parsing_errors=True,
#         #verbose=True
#     )
# st_cb = StdOutCallbackHandler()
# print(agent.run(input="how is the weather in wismar ?", callbacks=[st_cb]))
################################################################################
agent = create_openai_tools_agent(tools=tools, llm=llm, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "what is the date today?"})
