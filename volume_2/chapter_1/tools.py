"""Local deterministic tools for LLM agents."""

from __future__ import annotations

import ast
import math
import re
import uuid
from typing import Any, Callable, Dict

from dateutil import parser as date_parser


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


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


def generate_uuid(_: Any = None) -> Dict[str, Any]:
    """Generate a unique UUID identifier."""
    return {
        "uuid": str(uuid.uuid4())
    }



TOOLS: Dict[str, Callable[[Any], Dict[str, Any]]] = {
    "calculator": calculator,
    "resolve_datetime": resolve_datetime,
    "generate_uuid": generate_uuid,
}


TOOL_DESCRIPTIONS: Dict[str, str] = {
    "calculator": "Safely evaluate arithmetic expressions.",
    "resolve_datetime": "Resolve date/time phrases into ISO and human-readable values.",
    "generate_uuid": "Generate a unique UUID identifier.",
}


DEFAULT_TOOL_PRIORITY = ["calculator", "resolve_datetime", "generate_uuid"]
DEFAULT_KEYWORD_ROUTES = {
    "calculator": ["calculate", "calc", "compute", "math", "equation", "percentage"],
    "resolve_datetime": ["resolve datetime", "parse date", "parse datetime", "when is", "tomorrow", "next week", "next month", "today"],
    "generate_uuid": ["uuid", "unique id", "ticket id", "identifier"],
}


def _trigger_match(prompt_l: str, trigger: str) -> bool:
    if " " in trigger or any(ch in trigger for ch in [":", "/", ".", "-"]):
        return trigger in prompt_l
    return re.search(rf"\b{re.escape(trigger)}\b", prompt_l) is not None


def route_tool_for_prompt(prompt: str, available_tool_names: list[str] | None = None) -> tuple[str, str] | None:
    """Route a prompt to a single deterministic local tool call when confidence is high."""
    text = (prompt or "").strip()
    if not text:
        return None

    available = set(available_tool_names or DEFAULT_TOOL_PRIORITY)

    calculator_match = re.match(r"^(?:calculate|calc|compute)\s+(.+)$", text, flags=re.IGNORECASE)
    if calculator_match and "calculator" in available:
        return ("calculator", calculator_match.group(1).strip())

    if any(op in text for op in ["+", "-", "*", "/", "="]) and "calculator" in available:
        return ("calculator", text.replace("=", " ").strip())

    datetime_match = re.match(
        r"^(?:resolve\s+datetime|parse\s+date(?:time)?|when\s+is)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if datetime_match and "resolve_datetime" in available:
        return ("resolve_datetime", datetime_match.group(1).strip())

    if any(token in text.lower() for token in ["tomorrow", "next week", "next month", "today", " at "]) and "resolve_datetime" in available:
        return ("resolve_datetime", text)

    uuid_match = re.match(
        r"^(?:generate|create|make)\s+(?:a\s+)?(?:unique\s+)?(?:uuid|id|identifier|ticket id|ticket identifier)\b.*$",
        text,
        flags=re.IGNORECASE,
    )
    if uuid_match and "generate_uuid" in available:
        return ("generate_uuid", "")

    if any(
        phrase in text.lower()
        for phrase in [
            "generate a unique id",
            "generate an id",
            "generate a uuid",
            "create a unique id",
            "create an id",
            "create a uuid",
            "new ticket id",
            "unique ticket id",
            "unique identifier",
        ]
    ) and "generate_uuid" in available:
        return ("generate_uuid", "")


    return None


def route_tools_for_prompt(prompt: str, available_tool_names: list[str] | None = None) -> list[str]:
    """Route a prompt to a minimal relevant tool set, with stable fallback ordering."""
    text = prompt or ""
    prompt_l = text.lower()
    available = available_tool_names or DEFAULT_TOOL_PRIORITY
    available_set = set(available)
    selected = set()

    for tool_name, triggers in DEFAULT_KEYWORD_ROUTES.items():
        if tool_name not in available_set:
            continue
        if any(_trigger_match(prompt_l, trigger) for trigger in triggers):
            selected.add(tool_name)

    has_math_expression = any(op in text for op in ["+", "*", "/", "="]) or (" - " in text)
    if has_math_expression and "calculator" in available_set:
        selected.add("calculator")

    if not selected:
        return [name for name in DEFAULT_TOOL_PRIORITY if name in available_set]

    return [name for name in DEFAULT_TOOL_PRIORITY if name in selected and name in available_set]


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
