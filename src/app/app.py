from fastapi import FastAPI,Response,status,Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_ollama.chat_models import ChatOllama
import pathlib
import sys
PROJECT_PATH = pathlib.Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0,str(PROJECT_PATH))
import time
from agent.agent import AssistantAgent
from typing import Union
import asyncio
from agent.mcp_client import get_client
import logging
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

llm = ChatOllama( model="gpt-oss:20b",
    temperature="0")

app = FastAPI(name="Agent Bot",debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
)
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
   
    def event_generator():
         for message_chunk,meta_data in agent.stream(
            {"messages": message},
            config=config,
            version="v1",
            stream_mode="messages"
        ):
            # print("message_chunk ",message_chunk)
            # Only stream model tokens
            token = message_chunk.content
            if token:
                yield token
                time.sleep(0.01)

    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
         headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
    # response = agent.stream({"messages": message}, config, stream_mode="values")
    # return response

# @app.get("/stream")
# def chat_assistant(message: str , user_id: str ):

#     config = {
#         "configurable": {"thread_id": user_id, "user_id": user_id}
#     }

#     def event_generator():
#         for event in agent.stream({"messages": message}, config=config, stream_mode="events"):
#             if event["event"] == "on_chat_model_stream":
#                 chunk = event["data"]["chunk"].content
#                 if chunk:
#                     yield chunk
#                     # force flush to send immediately
#                     time.sleep(0.01)

#     return StreamingResponse(
#         event_generator(),
#        media_type="text/event-stream",
#     )


async def init_agent():
    client = await get_client()
    tools = await client.get_tools()
    print("tools are ",tools)
    agent = await AssistantAgent( 
                        llm=llm,
                        conn_string="postgresql://postgres:root@localhost:5432/agent_bot",
                        tools=tools,
                        sync=True
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