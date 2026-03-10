#!/usr/bin/env python3

from langchain.agents import create_agent
from langchain.tools import tool

from tools import *


def log_llm(prompt):
    print("\n================ LLM PROMPT ================")
    print(prompt)
    print("===========================================\n")


def log_tool(name, fn):

    def wrapper(arg):

        print("\n------------- LOCAL TOOL CALL -------------")
        print("Tool:", name)
        print("Input:", arg)

        result = fn(arg)

        print("\n------------- TOOL RESULT -----------------")
        print(result)
        print("-------------------------------------------\n")

        return result

    return wrapper


@tool
def summarize_text_tool(text:str):
    return log_tool("summarize_text", summarize_text)(text)

@tool
def extract_keywords_tool(text:str):
    return log_tool("extract_keywords", extract_keywords)(text)

@tool
def extract_tasks_tool(text:str):
    return log_tool("extract_tasks", extract_tasks)(text)

@tool
def score_priority_tool(text:str):
    return log_tool("score_priority", score_priority)(text)

@tool
def route_workflow_tool(text:str):
    return log_tool("route_workflow", route_workflow)(text)

@tool
def parse_content_tool(content:str):
    return log_tool("parse_content", parse_content)(content)

@tool
def resolve_datetime_tool(text:str):
    return log_tool("resolve_datetime", resolve_datetime)(text)

@tool
def format_json_tool(input:str):
    return log_tool("format_json", format_json)(input)

@tool
def calculator_tool(expression:str):
    return log_tool("calculator", calculator)(expression)

@tool
def analyze_text_tool(text:str):
    return log_tool("analyze_text", analyze_text)(text)



agent = create_agent(
    model="gpt-4.1",
    tools=[ ... ],
    system_prompt="""
You are an AI assistant that can use tools.

Use the format:

Thought:
Action:
Action Input:
Observation:

Repeat until finished.

Final Answer:
"""
)

print("\n===== LangChain Agent CLI =====\n")

while True:

    try:

        user_input = input("> ")

        print("\n============== USER INPUT ==============")
        print(user_input)
        print("========================================")

        log_llm(user_input)

        result = agent.invoke({
            "messages": [{"role": "user", "content": user_input}]
        })

        print("\n============= LLM RESPONSE =============")
        print(result["output"])
        print("========================================\n")

    except KeyboardInterrupt:
        print("\nExiting.")
        break
