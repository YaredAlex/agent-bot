from langchain_mcp_adapters.client import MultiServerMCPClient 
import asyncio
from langchain_core.tools import StructuredTool

async def get_client():
    client = MultiServerMCPClient({
        "local_tools":{
            "url":"http://localhost:3001/mcp",
            "transport":"streamable-http"
        }
    })
    tools = await client.get_tools()
    
    for tool in tools:
        print("tool name ",tool.name)

    # # invoking tool to check response
    # for tool in tools:
    #     print(await tool.ainvoke({"user_id":"id"}))

    return client

if __name__=="__main__":
    asyncio.run(get_client())