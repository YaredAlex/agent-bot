import os
from datetime import datetime
import uuid
from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from psycopg_pool import ConnectionPool
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,merge_message_runs
from langgraph.graph import MessagesState,START,END,StateGraph
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_ollama.chat_models import ChatOllama
from typing import TypedDict, Literal,Optional
from pydantic import BaseModel,Field
from trustcall import create_extractor
#Setting environment
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"]="My First App"
#CheckPointer
conn_string = "postgresql://postgres:root@localhost:5432/agent_bot"
# pool = ConnectionPool(conn_string, kwargs={"autocommit": True})
# memory = PostgresSaver(pool)
# store = PostgresStore(pool)


MODEL_SYSTEM_MESSAGE = """You are a helpful chatbot. 

You are designed to be a companion to a user, helping them to manage and assist shopping experiance.

You have a long term memory which keeps track of three things:
1. The user's profile (general information about them) 
2. The user's shopping history
3. General instructions for updating the preference or recommendation

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here is the User shopping histry (may be empty if no purchase have been added yet):
<history>
{history}
</history>

Here are the current user-specified preferences for updating the preference list (may be empty if no preferences have been specified yet):
<instructions>
{instructions}
</instructions>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below. 

2. Decide whether any of the your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling update_profile tool
- If needed update user history when updating profile to personalize conversation

3. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile
- Do not tell the user that you have updated instructions

4. Err on the side of updating profile or history. No need to ask for explicit permission.

5. Respond naturally to user user after a tool call was made to save memories, or if no tool call was made."""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user.

System Time: {time}"""


class AssistantAgent:
    def __init__(self,llm=None,tools=[],conn_string=None):

        if llm==None:
            raise ValueError("llm can not be None")
        if conn_string==None:
            print("Warrning conn_string is None message persist Inmemory")

        self.conn_string = conn_string or "postgresql://postgres:root@localhost:5432/agent_bot"
        if self.conn_string:
            self.pool = ConnectionPool(conn_string, kwargs={"autocommit": True})
            self.memory = PostgresSaver(self.pool)
            self.store = PostgresStore(self.pool)
        else:
            self.memory = InMemorySaver()
            self.store = InMemoryStore()
        #  NOTE: you need to call .setup() the first time you're using your checkpointer
        self.memory.setup()
        self.store.setup()
        # binding tool
        self.llm = llm
        self.tools = tools + AgentMemoryTools(memory=self.memory,store=self.store,llm=self.llm).get_tools()
        self.llm_with_tools = self._bind_tools(self.tools)

    def _bind_tools(self,tools):
        assert isinstance(tools,list), "List of tools is required!"
        return self.llm.bind_tools(tools)
    def _set_conn_string(self,conn_string):
        self.conn_string = conn_string
    
    
    def _build_graph(self):
       
        # Node definitions
        def assistant(state: MessagesState, config: RunnableConfig, store: BaseStore):
            """Load memories from the store and use them to personalize the chatbot's response."""
            # Get the user ID from the config
            user_id = config["configurable"]["user_id"]

            # Retrieve profile memory from the store
            namespace = ("profile", user_id)
            memories = store.search(namespace)
            if memories:
                user_profile = memories[0].value
            else:
                user_profile = None
            print("user profile ",user_profile)
            # retrive history
            namespace = ("history", user_id)
            memories = store.search(namespace)
            history = "\n".join(f"{mem.value}" for mem in memories)
            print("user history ",history)
            # Retrieve custom instructions
            namespace = ("instructions", user_id)
            memories = store.search(namespace)
            if memories:
                instructions = memories[0].value
            else:
                instructions = ""
            
            system_msg = MODEL_SYSTEM_MESSAGE.format(user_profile=user_profile, history=history, instructions=instructions)
            # Respond using memory as well as the chat history
            response = self.llm_with_tools.invoke([SystemMessage(content=system_msg)]+state["messages"])

            return {"messages": [response]}
        
        profile_extractor = create_extractor(
            self.llm,
            tools=[Profile],
            tool_choice="Profile",
        )
        
        def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
            """Reflect on the chat history and update the memory collection."""
            
            # Get the user ID from the config
            user_id = config["configurable"]["user_id"]

            # Define the namespace for the memories
            namespace = ("profile", user_id)
            # Retrieve the most recent memories for context
            existing_items = store.search(namespace)
            print("Exisiting_items ", existing_items)
            # Format the existing memories for the Trustcall extractor
            tool_name = "Profile"
            existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                                for existing_item in existing_items]
                                if existing_items
                                else None
                                )
            print("Exisiting memories ",existing_memories)
            # Merge the chat history and the instruction
            TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
            updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

            # Invoke the extractor
            result = profile_extractor.invoke({"messages": updated_messages, 
                                                "existing": existing_memories})
            print("result of profile extractor ",result)
            # Save the memories from Trustcall to the store
            for r, rmeta in zip(result["responses"], result["response_metadata"]):
                store.put(namespace,
                        rmeta.get("json_doc_id", str(uuid.uuid4())),
                        r.model_dump(mode="json"),
                    )
            tool_calls = state['messages'][-1].tool_calls
            return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id":tool_calls[0]['id']}]}
        
        
        def update_history(state: MessagesState, config: RunnableConfig, store: BaseStore):
            """Summarize recent conversation and persist rolling history summary."""

            user_id = config["configurable"]["user_id"]
            namespace = ("history", user_id)

            #  Get existing history summary
            existing_items = store.search(namespace)
            existing_summary = existing_items[0].value if existing_items else ""

            # Prepare conversation text (exclude system messages)
            conversation_text = "\n".join(
                f"{m.type.upper()}: {m.content}"
                for m in state["messages"]
                if hasattr(m, "content")
            )

            # Create summarization prompt
            summary_prompt = f"""
                        You are a conversation memory summarizer.
                        Existing summary:
                        {existing_summary}
                        New conversation:
                        {conversation_text}
                        Update the summary to include important long-term context about the user,
                        their preferences, goals, interests, and important discussion points.
                        Keep it concise but informative.
                        """
            print("summary prompt ",summary_prompt)
            summary_response = self.llm.invoke(summary_prompt)

            updated_summary = summary_response.content

            # Save updated summary (overwrite old one)
            store.put(
                namespace,
                "rolling_summary",  # fixed key to overwrite
                updated_summary
            )

            return "history updated"

        
        #Node for routing messages
        def route_message(state: MessagesState,) -> Literal[END,"tools", "update_profile", "update_history" ]:

            last_message = state["messages"][-1]

            # No tool calls 
            if not getattr(last_message, "tool_calls", None):
                return END

            tool_call = last_message.tool_calls[0]
            tool_name = tool_call["name"]

            # Manual memory nodes
            if tool_name == "update_profile":
                return "update_profile"

            elif tool_name == "update_history":
                return "update_history"

            # elif tool_name == "update_instructions":
            #     return "update_instructions"

            # Everything else (MCP tools)
            else:
                return "tools"
            

        #build the graph here
        self.builder = StateGraph(MessagesState)
        self.builder.add_node("assistant",assistant)
        self.builder.add_node("tools",ToolNode(self.tools))
        self.builder.add_node("update_profile",update_profile)
        self.builder.add_node("update_history",update_history)
        #adding edge
        self.builder.add_edge(START,"assistant")
        self.builder.add_conditional_edges("assistant",route_message)
        self.builder.add_edge("update_profile","assistant")
        self.builder.add_edge("update_history","assistant")
        self.builder.add_edge("tools","assistant")
        self.graph = self.builder.compile(checkpointer=self.memory,store=self.store)
        return self.graph
    
    def get_graph(self):
        return self._build_graph()
    

####
#Tools
####
# User profile schema
class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default_factory=list
    )
    interests: list[str] = Field(
        description="Interests that the user has", 
        default_factory=list
    )
class AgentMemoryTools:
    def __init__(self,memory=None,store=None,llm=None):
        assert memory!=None or store!=None,"Memory and Store are None"
        assert llm!=None, "Warning agentMemory needs LLM model"
        self.memory = memory
        self.store = store
        self.llm = llm

    def get_tools(self):
        # Create the Trustcall extractor for updating the user profile 
        def update_profile():
            """
            Docstring for update_profile
            to update user profile based on preference
            """
            return "update_profile"
        def update_history():
            """
            Docstring for update_history
            update conversation history with summary
            """
        
        return [update_history,update_profile]



# testing react-agent
llm = ChatOllama( model="gpt-oss:20b",
    temperature="0")
# initializing graph
agent = AssistantAgent( 
    llm=llm,
    conn_string=conn_string,
    tools=[],
).get_graph()

config = {"configurable": {"thread_id": "user_id_1", "user_id": "demo_user"}}

for chunk in agent.stream({"messages": "could you summarize our conversation?"}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
