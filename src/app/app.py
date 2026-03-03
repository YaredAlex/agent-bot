import sys
import asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import pathlib
PROJECT_PATH = pathlib.Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0,str(PROJECT_PATH))
from fastapi import FastAPI,Response,status,Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_ollama.chat_models import ChatOllama
from agent.agent import AssistantAgent
from typing import Union
from agent.mcp_client import get_client
import logging
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
async def chat_assistant(message:str=Form(...),user_id:Union[str,int]=Form(...)):
    global agent
    config = {"configurable": {"thread_id": user_id, "user_id": user_id}}
    if (agent==None):
        print("getting agent ")
        agent = await init_agent()
    async def event_generator():
         async for message_chunk,meta_data in agent.astream(
            {"messages": message},
            config=config,
            version="v1",
            stream_mode="messages"
        ):
            is_tool = getattr(message_chunk, "is_tool_call", False)  # if your chunk has this attribute
            # Or use meta_data flag if available
            if meta_data.get("tool_name") or is_tool:
                # skip streaming tool outputs
                continue
            token = message_chunk.content
            if isinstance(token, list):
            # join list items into a single string
                token = " ".join(str(c) for c in token)
            if token:
                yield token
                await asyncio.sleep(0.01)

    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        #  headers={
        #     "Cache-Control": "no-cache",
        #     "Connection": "keep-alive",
        # }
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
    agent = await AssistantAgent( 
                        llm=llm,
                        conn_string="postgresql://postgres:root@localhost:5432/agent_bot",
                        tools=tools,
                        sync=False
                        ).get_graph()
    return agent

if __name__=="__main__":
    #check connection for mcp server if server is down warn user of exit app
    
    try:
       agent = asyncio.run(init_agent())
    except Exception as e:
        logging.error("Faild when initializing agent ",e)
    import uvicorn
    # important on windows asyncio:SelectorEventLoop
    uvicorn.run("app:app",loop="asyncio:SelectorEventLoop",host="127.0.0.1",port=4000,reload=False)