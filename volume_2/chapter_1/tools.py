"""Local deterministic tools for LLM agents."""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Callable, Dict

from dateutil import parser as date_parser


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _safe_parse_json_or_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def resolve_datetime(text: str) -> Dict[str, Any]:
    """Parses datetime-like input into ISO and human-readable formats."""
    text = _normalize_whitespace(text)
    if not text:
        return {"error": "No datetime text provided."}

    try:
        dt = date_parser.parse(text, fuzzy=True)
    except (ValueError, OverflowError) as exc:
        return {"error": f"Could not parse datetime: {exc}"}

    return {
        "original": text,
        "resolved_iso": dt.isoformat(),
        "human_readable": dt.strftime("%A, %B %d, %Y %I:%M %p"),
    }


def format_json(value: Any) -> Dict[str, Any]:
    """Pretty-formats JSON-compatible input."""
    parsed = _safe_parse_json_or_string(value)
    try:
        formatted = json.dumps(parsed, indent=2, ensure_ascii=False, sort_keys=True)
        return {"formatted_json": formatted}
    except (TypeError, ValueError) as exc:
        return {"error": f"Could not format JSON: {exc}"}


class _SafeMathEvaluator(ast.NodeVisitor):
    ALLOWED_BINOPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
    }
    ALLOWED_UNARYOPS = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }
    ALLOWED_NAMES = {
        "pi": math.pi,
        "e": math.e,
    }
    ALLOWED_FUNCS = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "fabs": math.fabs,
        "ceil": math.ceil,
        "floor": math.floor,
    }

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")

    def visit_Num(self, node: ast.Num) -> Any:
        return node.n

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        operator = type(node.op)
        if operator not in self.ALLOWED_BINOPS:
            raise ValueError(f"Unsupported operator: {operator.__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self.ALLOWED_BINOPS[operator](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operator = type(node.op)
        if operator not in self.ALLOWED_UNARYOPS:
            raise ValueError(f"Unsupported unary operator: {operator.__name__}")
        operand = self.visit(node.operand)
        return self.ALLOWED_UNARYOPS[operator](operand)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.ALLOWED_NAMES:
            return self.ALLOWED_NAMES[node.id]
        raise ValueError(f"Unsupported name: {node.id}")

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed.")
        func_name = node.func.id
        if func_name not in self.ALLOWED_FUNCS:
            raise ValueError(f"Unsupported function: {func_name}")
        args = [self.visit(arg) for arg in node.args]
        return self.ALLOWED_FUNCS[func_name](*args)

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Unsupported expression: {type(node).__name__}")


def calculator(expression: str) -> Dict[str, Any]:
    """Safely evaluates arithmetic expressions."""
    expression = _normalize_whitespace(expression)
    if not expression:
        return {"error": "No expression provided."}

    try:
        tree = ast.parse(expression, mode="eval")
        evaluator = _SafeMathEvaluator()
        result = evaluator.visit(tree)
        return {
            "expression": expression,
            "result": result,
        }
    except Exception as exc:
        return {"error": f"Could not evaluate expression: {exc}"}


TOOLS: Dict[str, Callable[[Any], Dict[str, Any]]] = {
    "calculator": calculator,
    "resolve_datetime": resolve_datetime,
    "format_json": format_json,
}

TOOL_DESCRIPTIONS: Dict[str, str] = {
    "calculator": "Safely evaluate arithmetic expressions.",
    "resolve_datetime": "Resolve date/time phrases into ISO and human-readable values.",
    "format_json": "Pretty-format JSON-compatible input.",
}


def list_tools() -> Dict[str, Any]:
    return {
        "tools": [
            {"name": name, "description": TOOL_DESCRIPTIONS[name]}
            for name in TOOLS
        ]
    }


def build_tools_prompt() -> str:
    lines = [
        "You can use the following local deterministic tools:",
        "",
    ]
    for name, description in TOOL_DESCRIPTIONS.items():
        lines.append(f"- {name}: {description}")

    lines.extend([
        "",
        "When a tool is needed, respond in this format:",
        "TOOL: <tool_name>",
        "INPUT: <tool_input>",
        "",
        "After receiving the tool result, continue your reasoning using the observation.",
    ])
    return "\n".join(lines)


def run_tool(name: str, input_data: Any) -> Dict[str, Any]:
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"Tool '{name}' not found."}

    try:
        return tool(input_data)
    except Exception as exc:
        return {"error": f"Tool '{name}' failed: {exc}"}
