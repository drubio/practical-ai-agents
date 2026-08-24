import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).resolve().parent / "../../shared/.env"
load_dotenv(dotenv_path=env_path)

# LangChain and LangGraph imports
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Define Tool
@tool
def generate_uuid_tool():
    """Generate a unique UUID identifier."""
    return {
        "uuid": str(uuid.uuid4())
    }

tools = [generate_uuid_tool]
llm = ChatOpenAI(model="gpt-5.6-luna", reasoning_effort="none").bind_tools(tools)

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

class CustomAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_call_count: int
    user: str

"""
# Equivalent state using inheritance
class CustomAgentState(MessagesState):
    llm_call_count: int
    user: str
"""

# Node function to unpack state["messages"] for the LLM
def call_model(state: CustomAgentState):
    
    response = llm.invoke(state["messages"])
    # Increase llm_call_count after llm.invoke

    # Safely get llm_call_count in case it was not initialzed
    current_count = state.get("llm_call_count", 0)
    new_count = current_count + 1
        
    print(f"Running LLM on behalf of {state['user']}")
    print(f"LLM has been run {new_count} times")
    return {
        "messages": [response],
        "llm_call_count": new_count
    }

# Construct Graph
builder = StateGraph(CustomAgentState)

builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")
builder.add_edge("model", END)


graph = builder.compile()


# Invoke Graph
result = graph.invoke(
    {
        "messages": [
            SystemMessage(
                content="Use generate_uuid when user asks for UUID. Keep responses short."
            ),
            HumanMessage(
                content="Generate a ticket ID for last night's network outage"
            ),
        ],
        "llm_call_count":0,        
        "user":"drubio"
    }
)

print(result["messages"][-1].content)

