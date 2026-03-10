"""
Local deterministic tools for LLM agents.

These tools are designed to integrate with LangChain or LlamaIndex agents
while remaining framework-neutral. They provide reliable, non-LLM utilities
for:

- summarization
- keyword extraction
- task extraction
- priority scoring
- workflow routing
- HTML/text parsing
- datetime resolution
- JSON formatting
- calculations

The JavaScript counterpart is implemented in tools.js with equivalent names
using camelCase.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from sklearn.feature_extraction.text import TfidfVectorizer


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "if", "in", "into", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "she", "so", "that", "the",
    "their", "them", "they", "this", "to", "us", "was", "we", "were", "will",
    "with", "you", "your"
}

TASK_PATTERNS = [
    r"\bTODO\b[:\-]?\s*(.+)",
    r"\baction item\b[:\-]?\s*(.+)",
    r"\bneed to\b\s+(.+)",
    r"\bmust\b\s+(.+)",
    r"\bshould\b\s+(.+)",
    r"\bplease\b\s+(.+)",
]

HIGH_PRIORITY_HINTS = {
    "urgent", "asap", "immediately", "blocker", "critical", "production",
    "outage", "security", "deadline", "today", "now", "failure", "broken"
}
MEDIUM_PRIORITY_HINTS = {
    "soon", "review", "follow up", "follow-up", "important", "tomorrow",
    "next", "schedule", "plan", "pending"
}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _split_sentences(text: str) -> List[str]:
    text = _normalize_whitespace(text)
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


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


def summarize_text(text: str, max_sentences: int = 3) -> Dict[str, Any]:
    """
    Deterministic extractive summarizer using TF-IDF sentence scoring.
    """
    text = _normalize_whitespace(text)
    if not text:
        return {"summary": "", "sentence_count": 0}

    sentences = _split_sentences(text)
    if not sentences:
        return {"summary": text, "sentence_count": 1}

    if len(sentences) <= max_sentences:
        return {
            "summary": " ".join(sentences),
            "sentence_count": len(sentences),
        }

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(sentences)
        scores = matrix.sum(axis=1).A1
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )[:max_sentences]
        top_indices = sorted(index for index, _ in ranked)
        summary = " ".join(sentences[i] for i in top_indices)
    except Exception:
        summary = " ".join(sentences[:max_sentences])

    return {
        "summary": summary,
        "sentence_count": len(sentences),
    }


def extract_keywords(text: str, top_k: int = 8) -> Dict[str, Any]:
    """
    Keyword extraction using TF-IDF over the document itself.
    """
    text = _normalize_whitespace(text)
    if not text:
        return {"keywords": []}

    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = matrix.toarray()[0]
        ranked = sorted(
            zip(feature_names, scores),
            key=lambda item: item[1],
            reverse=True
        )
        keywords = [term for term, score in ranked if score > 0][:top_k]
    except Exception:
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", text.lower())
        counts = Counter(word for word in words if word not in STOPWORDS)
        keywords = [word for word, _ in counts.most_common(top_k)]

    return {"keywords": keywords}


def extract_tasks(text: str) -> Dict[str, Any]:
    """
    Extracts likely tasks/action items from text using simple heuristics.
    """
    text = _normalize_whitespace(text)
    if not text:
        return {"tasks": []}

    sentences = _split_sentences(text)
    tasks: List[Dict[str, Any]] = []

    for sentence in sentences:
        lowered = sentence.lower()

        direct_match = False
        for pattern in TASK_PATTERNS:
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if match:
                task_text = _normalize_whitespace(match.group(1))
                if task_text:
                    tasks.append({
                        "task": task_text.rstrip("."),
                        "source": sentence,
                    })
                    direct_match = True
                break

        if direct_match:
            continue

        if any(
            token in lowered
            for token in ["fix", "implement", "review", "update", "send", "draft", "prepare", "schedule"]
        ):
            tasks.append({
                "task": sentence.rstrip("."),
                "source": sentence,
            })

    deduped = []
    seen = set()
    for item in tasks:
        key = item["task"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)

    return {"tasks": deduped}


def score_priority(text: str) -> Dict[str, Any]:
    """
    Deterministic priority scoring based on urgency/blocker cues.
    """
    text = _normalize_whitespace(text).lower()
    if not text:
        return {"priority": "low", "score": 0, "reasons": []}

    score = 0
    reasons: List[str] = []

    for hint in HIGH_PRIORITY_HINTS:
        if hint in text:
            score += 3
            reasons.append(f"contains high-priority cue: {hint}")

    for hint in MEDIUM_PRIORITY_HINTS:
        if hint in text:
            score += 1
            reasons.append(f"contains medium-priority cue: {hint}")

    if re.search(r"\b\d{1,2}:\d{2}\b", text):
        score += 1
        reasons.append("contains explicit time")

    if re.search(r"\b(today|tomorrow|friday|monday|tuesday|wednesday|thursday|saturday|sunday)\b", text):
        score += 1
        reasons.append("contains schedule cue")

    if score >= 6:
        priority = "high"
    elif score >= 2:
        priority = "medium"
    else:
        priority = "low"

    return {
        "priority": priority,
        "score": score,
        "reasons": reasons,
    }


def route_workflow(text: str) -> Dict[str, Any]:
    """
    Chooses a likely workflow bucket for the input.
    """
    text = _normalize_whitespace(text).lower()

    routing_rules: List[Tuple[str, Tuple[str, ...]]] = [
        ("incident_response", ("outage", "incident", "broken", "failure", "production", "down")),
        ("task_planning", ("todo", "action item", "plan", "task", "milestone", "deliverable")),
        ("communication_draft", ("reply", "email", "respond", "draft", "message")),
        ("meeting_followup", ("meeting", "notes", "follow up", "follow-up", "summary")),
        ("documentation", ("docs", "documentation", "guide", "readme", "manual")),
        ("general_analysis", tuple()),
    ]

    for route, keywords in routing_rules:
        if route == "general_analysis":
            return {"route": route}
        if any(keyword in text for keyword in keywords):
            return {"route": route}

    return {"route": "general_analysis"}


def parse_content(content: str) -> Dict[str, Any]:
    """
    Parses HTML or plain text into normalized content sections.
    """
    content = content or ""
    soup = BeautifulSoup(content, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else None

    headings = [
        element.get_text(" ", strip=True)
        for element in soup.find_all(["h1", "h2", "h3"])
    ]

    links = []
    for anchor in soup.find_all("a", href=True):
        links.append({
            "text": anchor.get_text(" ", strip=True),
            "href": anchor["href"],
        })

    plain_text = _normalize_whitespace(soup.get_text(" ", strip=True))
    if not plain_text:
        plain_text = _normalize_whitespace(content)

    sections = []
    for heading in soup.find_all(["h1", "h2", "h3"]):
        section_text_parts = []
        sibling = heading.find_next_sibling()
        while sibling and sibling.name not in {"h1", "h2", "h3"}:
            section_text_parts.append(sibling.get_text(" ", strip=True))
            sibling = sibling.find_next_sibling()

        sections.append({
            "heading": heading.get_text(" ", strip=True),
            "content": _normalize_whitespace(" ".join(section_text_parts)),
        })

    return {
        "title": title,
        "headings": headings,
        "links": links,
        "plain_text": plain_text,
        "sections": sections,
    }


def resolve_datetime(text: str) -> Dict[str, Any]:
    """
    Parses datetime-like input into ISO and human-readable formats.
    """
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
    """
    Pretty-formats JSON-compatible input.
    """
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
        ast.Pow: lambda a, b: a ** b,
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

    def visit(self, node: ast.AST) -> Any:
        return super().visit(node)

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
    """
    Safely evaluates arithmetic expressions.
    """
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


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Composite tool that summarizes text, extracts keywords/tasks,
    scores priority, and proposes next steps.
    """
    text = _normalize_whitespace(text)
    if not text:
        return {
            "summary": "",
            "keywords": [],
            "tasks": [],
            "priority": {"priority": "low", "score": 0, "reasons": []},
            "route": "general_analysis",
            "next_steps": [],
        }

    summary_result = summarize_text(text)
    keywords_result = extract_keywords(text)
    tasks_result = extract_tasks(text)
    priority_result = score_priority(text)
    route_result = route_workflow(text)

    next_steps: List[str] = []
    if tasks_result["tasks"]:
        next_steps.extend(
            f"Complete task: {task['task']}"
            for task in tasks_result["tasks"][:3]
        )
    else:
        next_steps.append("Review the summary and confirm the intended action.")
        next_steps.append("Clarify ownership and deadline if missing.")

    return {
        "summary": summary_result["summary"],
        "keywords": keywords_result["keywords"],
        "tasks": tasks_result["tasks"],
        "priority": priority_result,
        "route": route_result["route"],
        "next_steps": next_steps,
    }


TOOLS: Dict[str, Callable[[Any], Dict[str, Any]]] = {
    "summarize_text": summarize_text,
    "extract_keywords": extract_keywords,
    "extract_tasks": extract_tasks,
    "score_priority": score_priority,
    "route_workflow": route_workflow,
    "parse_content": parse_content,
    "resolve_datetime": resolve_datetime,
    "format_json": format_json,
    "calculator": calculator,
    "analyze_text": analyze_text,
}


TOOL_DESCRIPTIONS: Dict[str, str] = {
    "summarize_text": "Create a short deterministic summary of input text.",
    "extract_keywords": "Extract top keywords from input text.",
    "extract_tasks": "Extract likely action items or tasks from text.",
    "score_priority": "Estimate priority level from urgency and blocker cues.",
    "route_workflow": "Route input into a likely workflow category.",
    "parse_content": "Parse HTML or text into normalized sections, headings, links, and plain text.",
    "resolve_datetime": "Resolve date/time phrases into ISO and human-readable values.",
    "format_json": "Pretty-format JSON-compatible input.",
    "calculator": "Safely evaluate arithmetic expressions.",
    "analyze_text": "Run a combined text analysis including summary, keywords, tasks, priority, and next steps.",
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
        "After receiving the tool result, continue your reasoning using the observation."
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
