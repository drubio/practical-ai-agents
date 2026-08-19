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
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


# Define Tool
@tool
def generate_uuid_tool():
    """Generate a unique UUID identifier."""
    return {
        "uuid": str(uuid.uuid4())
    }

tools = [generate_uuid_tool]
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)


# Node function to unpack state["messages"] for the LLM
def call_agent(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Construct Graph
builder = StateGraph(MessagesState)

builder.add_node("agent", call_agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

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
        ]
    }
)

print(result["messages"][-1].content)
