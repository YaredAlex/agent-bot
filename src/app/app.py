from fastapi import FastAPI
from langchain_ollama.chat_models import ChatOllama
from agent.agent import AssistantAgent
from typing import Union

from agent.mcp_client import get_client

llm = ChatOllama( model="gpt-oss:20b",
    temperature="0")

app = FastAPI("Agent Bot",debug=True)

@app.get("/health")
def get_health():

    return ""

@app.post("/")
def chat_assistant(message:str,user_id:Union[str,int]):
    config = {"configurable": {"thread_id": user_id, "user_id": user_id}}
    response = agent.stream({"messages": message}, config, stream_mode="values")
    return response


if __name__=="__main__":
    #check connection for mcp server if server is down warn user of exit app
    client = get_client()
    tools = await client.get_tools()
    agent = AssistantAgent( 
                        llm=llm,
                        conn_string="postgresql://postgres:root@localhost:5432/agent_bot",
                        tools=tools,
                        ).get_graph()
    import uvicorn
    uvicorn.run(app=app,host="127.0.0.1",port=4000)