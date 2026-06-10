"""Local tools used by LangChain agent exercises."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field


def calculator(expression: str) -> Dict[str, Any]:
    text = (expression or "").strip()
    if not text:
        return {"error": "No expression provided."}

    if any(ch not in "0123456789+-*/().% ^" for ch in text):
        return {"error": "Only basic arithmetic characters are allowed."}

    try:
        result = eval(text.replace("^", "**"), {"__builtins__": {}}, {})
        if not isinstance(result, (int, float)):
            return {"error": "Invalid numeric result."}
        return {"expression": text, "result": result}
    except Exception as exc:
        return {"error": f"Could not evaluate expression: {exc}"}


def resolve_datetime(text: str) -> Dict[str, Any]:
    value = (text or "").strip()
    if not value:
        return {"error": "No datetime text provided."}

    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return {"error": "Could not parse datetime."}

    return {
        "original": value,
        "resolved_iso": dt.isoformat(),
        "human_readable": dt.strftime("%a, %d %b %Y %H:%M:%S"),
    }


def generate_uuid(_: Any = None) -> Dict[str, Any]:
    return {"uuid": str(uuid.uuid4())}


class CalculatorInput(BaseModel):
    """Schema for calculator calls that require an expression argument."""

    expression: str = Field(description="Arithmetic expression to evaluate.")


GENERATE_UUID_ARGS_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def create_calculator_tool(pending_tool_logs: list[dict[str, Any]]):
    """Create a LangChain calculator tool that records expression input."""

    from langchain_core.tools import tool

    @tool("calculator", args_schema=CalculatorInput)
    def calculator_tool(expression: str):
        """Evaluate a basic arithmetic expression."""
        output = calculator(expression)
        pending_tool_logs.append(
            {
                "name": "calculator",
                "input": {"expression": expression},
                "output": output,
            }
        )
        return output

    return calculator_tool


def create_generate_uuid_tool(pending_tool_logs: list[dict[str, Any]]):
    """Create a LangChain tool that records each generated UUID call."""

    from langchain_core.tools import tool

    @tool("generate_uuid", args_schema=GENERATE_UUID_ARGS_SCHEMA)
    def generate_uuid_tool():
        """Generate a unique UUID identifier."""
        output = generate_uuid()
        pending_tool_logs.append(
            {"name": "generate_uuid", "input": {}, "output": output}
        )
        return output

    return generate_uuid_tool
