from fastapi import FastAPI,Response,status,Form
from fastapi.responses import StreamingResponse
from langchain_ollama.chat_models import ChatOllama
import pathlib
import sys
PROJECT_PATH = pathlib.Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0,str(PROJECT_PATH))
from agent.agent import AssistantAgent
from typing import Union
import asyncio
from agent.mcp_client import get_client
import logging
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

llm = ChatOllama( model="gpt-oss:20b",
    temperature="0")

app = FastAPI(name="Agent Bot",debug=True)
agent = None
@app.get("/health")
def get_health(response:Response):
    global agent
    print("health agent ",agent)
    if agent!=None:
        return {"message":"Model is Online!"}
    response.status_code = status.HTTP_409_CONFLICT
    return {"message":"Model is offline"}

@app.post("/")
def chat_assistant(message:str=Form(...),user_id:Union[str,int]=Form(...)):
    config = {"configurable": {"thread_id": user_id, "user_id": user_id}}
   
    async def event_generator():
        async for event in agent.astream_events(
            {"messages": message},
            config=config,
            version="v1",
        ):

            # Only stream model tokens
            if event["event"] == "on_chat_model_stream":

                token = event["data"]["chunk"].content

                if token:
                    yield token

    return StreamingResponse(
        event_generator(),
        media_type="text/plain"
    )
    # response = agent.stream({"messages": message}, config, stream_mode="values")
    # return response


async def init_agent():
    client = await get_client()
    tools = await client.get_tools()
    print("tools are ",tools)
    agent = AssistantAgent( 
                        llm=llm,
                        conn_string="postgresql://postgres:root@localhost:5432/agent_bot",
                        tools=tools,
                        ).get_graph()
    print("agent is created")
    return agent

if __name__=="__main__":
    #check connection for mcp server if server is down warn user of exit app
    
    try:
       agent = asyncio.run(init_agent())
    except Exception as e:
        logging.error("Faild when initializing agent ",e)

    import uvicorn
    uvicorn.run(app=app,host="127.0.0.1",port=4000)