"""Local tools used by LangChain agent exercises."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict


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
