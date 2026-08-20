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


# Node function to unpack state["messages"] for the LLM
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Construct Graph
builder = StateGraph(MessagesState)

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
        ]
    }
)

print(result["messages"][-1].content)
