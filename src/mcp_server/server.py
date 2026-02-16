# demo mcp server
from mcp.server.fastmcp import FastMCP 

mcp = FastMCP("local_tools",port="3001")

@mcp.tool()
def get_user_info(user_id:str):
    """
    method for fetching user profile using user id
    
    :param user_id: Description
    :type user_id: str
    """
    return "user id is random"

@mcp.tool()
def get_user_history(user_id:str):
    """
    method for getting user conversation history
    
    :param user_id: Description
    :type user_id: str
    """

    return "history"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")