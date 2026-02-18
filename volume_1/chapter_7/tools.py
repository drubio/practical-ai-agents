"""
Chapter 7 tool utilities (Python).

Tools in this chapter are intentionally:
- framework-agnostic (LangChain/LlamaIndex),
- easy to port to JS,
- API-key-free where possible.

New in this version:
- Wikipedia/Wikimedia "evidence + media pack" tool for ANY query.
- Optional DuckDuckGo Instant Answer tool (no-key, limited).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import re
import urllib.parse
import urllib.request

import markdown
import pytz


@dataclass(frozen=True)
class ToolDefinition:
    """Serializable metadata that can be shared with prompt templates."""
    name: str
    description: str
    parameters: Dict[str, str]


ToolHandler = Callable[[Dict[str, Any]], str]


# -------------------------
# Existing tools
# -------------------------

def format_markdown_to_html(text: str) -> str:
    """Convert Markdown text to HTML."""
    return markdown.markdown(text or "")


def get_datetime(timezone: str = "UTC") -> str:
    """Return current datetime in the requested timezone."""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception as exc:
        return f"Error: {exc}"


# -------------------------
# HTTP helper (stdlib-only)
# -------------------------

def _http_get_json(url: str, timeout_s: int = 10) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "software-with-llms-ch7/1.0 (tool demo; https://example.invalid)"
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


# -------------------------
# Wikipedia / Wikimedia tools (no API keys)
# -------------------------

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_REST = "https://en.wikipedia.org/api/rest_v1"
_COMMONS_API = "https://commons.wikimedia.org/w/api.php"


def _wiki_opensearch(query: str, limit: int = 1) -> List[Tuple[str, str]]:
    """
    Very simple title resolver. Returns list of (title, url).
    Uses action=opensearch (no key).
    """
    q = urllib.parse.quote(query)
    url = f"{_WIKI_API}?action=opensearch&search={q}&limit={int(limit)}&namespace=0&format=json&origin=*"
    data = _http_get_json(url)
    # Format: [searchterm, [titles], [descriptions], [urls]]
    titles = data[1] if len(data) > 1 else []
    urls = data[3] if len(data) > 3 else []
    results: List[Tuple[str, str]] = []
    for i in range(min(len(titles), len(urls))):
        results.append((str(titles[i]), str(urls[i])))
    return results


def _wiki_page_summary(title: str) -> Dict[str, Any]:
    """
    Wikipedia REST summary endpoint (no key).
    Includes extract + thumbnail/originalimage when available.
    """
    safe_title = urllib.parse.quote(title.replace(" ", "_"))
    url = f"{_WIKI_REST}/page/summary/{safe_title}"
    return _http_get_json(url)


def _wiki_external_links(title: str, max_links: int = 10) -> List[str]:
    """
    Pull external links via action=parse&prop=externallinks (no key).
    """
    t = urllib.parse.quote(title)
    url = (
        f"{_WIKI_API}?action=parse&page={t}&prop=externallinks&format=json&origin=*"
    )
    data = _http_get_json(url)
    links = data.get("parse", {}).get("externallinks", []) or []
    cleaned: List[str] = []
    for link in links:
        if isinstance(link, str) and link.startswith("http"):
            cleaned.append(link)
        if len(cleaned) >= max_links:
            break
    return cleaned


def _wiki_page_images(title: str, max_files: int = 8) -> List[str]:
    """
    Get file titles used by the page (File:...), then resolve URLs via Commons imageinfo.
    Note: not all images are hosted on Commons, but many are; this is a best-effort demo.
    """
    t = urllib.parse.quote(title)
    url = (
        f"{_WIKI_API}?action=query&titles={t}"
        f"&prop=images&imlimit={int(max_files)}&format=json&origin=*"
    )
    data = _http_get_json(url)
    pages = (data.get("query", {}).get("pages", {}) or {}).values()
    file_titles: List[str] = []
    for page in pages:
        for img in (page.get("images", []) or []):
            name = img.get("title")
            if isinstance(name, str) and name.startswith("File:"):
                # filter to common media extensions
                if re.search(r"\.(png|jpg|jpeg|gif|svg|webm|ogv)$", name, re.I):
                    file_titles.append(name)
            if len(file_titles) >= max_files:
                break

    # Resolve file titles to URLs through Commons where possible
    resolved = _commons_resolve_file_urls(file_titles)
    return resolved


def _commons_resolve_file_urls(file_titles: List[str]) -> List[str]:
    if not file_titles:
        return []

    # Commons query titles are pipe-separated
    titles = "|".join(urllib.parse.quote(t) for t in file_titles[:20])
    url = (
        f"{_COMMONS_API}?action=query&titles={titles}"
        f"&prop=imageinfo&iiprop=url|mime&format=json&origin=*"
    )
    data = _http_get_json(url)
    out: List[str] = []
    pages = (data.get("query", {}).get("pages", {}) or {}).values()
    for page in pages:
        for ii in (page.get("imageinfo", []) or []):
            u = ii.get("url")
            if isinstance(u, str) and u.startswith("http"):
                out.append(u)
    # de-dupe preserving order
    seen = set()
    deduped: List[str] = []
    for u in out:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _commons_search_media(query: str, kind: str, limit: int = 6) -> List[Dict[str, str]]:
    """
    Best-effort Commons search for images/videos.
    - kind: "image" or "video"
    Uses list=search against Commons (no key).
    """
    q = query
    if kind == "video":
        # CirrusSearch often supports filetype:video
        q = f"{query} filetype:video"
    elif kind == "image":
        q = f"{query} filetype:bitmap"

    q_enc = urllib.parse.quote(q)
    url = (
        f"{_COMMONS_API}?action=query&list=search"
        f"&srsearch={q_enc}&srlimit={int(limit)}&format=json&origin=*"
    )
    data = _http_get_json(url)
    results = data.get("query", {}).get("search", []) or []

    file_titles: List[str] = []
    for r in results:
        title = r.get("title")
        # Commons file pages are titled like "File:Something.ext"
        if isinstance(title, str) and title.startswith("File:"):
            if kind == "video" and re.search(r"\.(webm|ogv)$", title, re.I):
                file_titles.append(title)
            elif kind == "image" and re.search(r"\.(png|jpg|jpeg|gif|svg)$", title, re.I):
                file_titles.append(title)

    urls = _commons_resolve_file_urls(file_titles)
    # Return both file page and direct url where we can
    out: List[Dict[str, str]] = []
    for ft, u in zip(file_titles, urls):
        out.append(
            {
                "file_title": ft,
                "file_page": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(ft.replace(' ', '_'))}",
                "url": u,
            }
        )
    return out


def get_wikipedia_evidence_pack(
    query: str,
    max_references: int = 8,
    max_page_media: int = 6,
    max_commons_images: int = 6,
    max_commons_videos: int = 4,
) -> Dict[str, Any]:
    """
    The "works for ANY query" tool.
    Returns a grounded pack of summary + sources + media without API keys.
    """
    query = (query or "").strip()
    if not query:
        return {"error": "Query is empty."}

    candidates = _wiki_opensearch(query, limit=1)
    if not candidates:
        return {"error": "No Wikipedia results found.", "query": query}

    title, page_url = candidates[0]
    summary_data = _wiki_page_summary(title)

    summary = summary_data.get("extract") or ""
    wiki_url = summary_data.get("content_urls", {}).get("desktop", {}).get("page") or page_url

    # main image(s)
    images: List[str] = []
    thumb = (summary_data.get("thumbnail", {}) or {}).get("source")
    orig = (summary_data.get("originalimage", {}) or {}).get("source")
    for u in [orig, thumb]:
        if isinstance(u, str) and u.startswith("http"):
            images.append(u)

    # extra images from page usage (best effort)
    page_images = _wiki_page_images(title, max_files=max_page_media)
    for u in page_images:
        if u not in images:
            images.append(u)

    # Commons searches for broader media coverage
    commons_images = _commons_search_media(query, kind="image", limit=max_commons_images)
    commons_videos = _commons_search_media(query, kind="video", limit=max_commons_videos)

    references = _wiki_external_links(title, max_links=max_references)

    return {
        "query": query,
        "topic": title,
        "page_url": wiki_url,
        "summary": summary,
        "references": references,
        "media": {
            "images": images[: max_page_media + 2],
            "commons_images": commons_images,
            "commons_videos": commons_videos,
        },
    }


# -------------------------
# DuckDuckGo Instant Answer (no key, limited)
# -------------------------

def duckduckgo_instant_answer(query: str, max_related: int = 10) -> Dict[str, Any]:
    """
    No-key endpoint, but NOT full search results.
    Good as a "quick fact" fallback or related-topic expander.
    """
    q = urllib.parse.quote((query or "").strip())
    url = f"https://api.duckduckgo.com/?q={q}&format=json&no_redirect=1&no_html=1"
    data = _http_get_json(url)

    related: List[Dict[str, str]] = []

    def _pull_related(items: Any) -> None:
        if not isinstance(items, list):
            return
        for it in items:
            if isinstance(it, dict) and "Topics" in it:
                _pull_related(it.get("Topics"))
            elif isinstance(it, dict):
                text = it.get("Text")
                first_url = it.get("FirstURL")
                if isinstance(text, str) and isinstance(first_url, str):
                    related.append({"text": text, "url": first_url})
            if len(related) >= max_related:
                return

    _pull_related(data.get("RelatedTopics"))

    return {
        "query": query,
        "abstract": data.get("AbstractText") or "",
        "abstract_url": data.get("AbstractURL") or "",
        "heading": data.get("Heading") or "",
        "related_topics": related[:max_related],
    }


# -------------------------
# Tool dispatch wrappers
# -------------------------

def _tool_format_markdown(parameters: Dict[str, Any]) -> str:
    return format_markdown_to_html(str(parameters.get("text", "")))


def _tool_get_datetime(parameters: Dict[str, Any]) -> str:
    return get_datetime(str(parameters.get("timezone", "UTC")))


def _tool_wikipedia_evidence_pack(parameters: Dict[str, Any]) -> str:
    query = str(parameters.get("query", "")).strip()
    max_references = int(parameters.get("max_references", 8))
    max_page_media = int(parameters.get("max_page_media", 6))
    max_commons_images = int(parameters.get("max_commons_images", 6))
    max_commons_videos = int(parameters.get("max_commons_videos", 4))
    pack = get_wikipedia_evidence_pack(
        query=query,
        max_references=max_references,
        max_page_media=max_page_media,
        max_commons_images=max_commons_images,
        max_commons_videos=max_commons_videos,
    )
    return _safe_json_dumps(pack)


def _tool_duckduckgo_instant_answer(parameters: Dict[str, Any]) -> str:
    query = str(parameters.get("query", "")).strip()
    max_related = int(parameters.get("max_related", 10))
    data = duckduckgo_instant_answer(query, max_related=max_related)
    return _safe_json_dumps(data)


TOOL_DEFINITIONS: List[ToolDefinition] = [
    ToolDefinition(
        name="format_markdown_to_html",
        description="Convert markdown text into HTML.",
        parameters={"text": "string - markdown content"},
    ),
    ToolDefinition(
        name="get_datetime",
        description="Get the current datetime in a timezone (e.g. UTC, Europe/Madrid).",
        parameters={"timezone": "string - IANA timezone"},
    ),
    ToolDefinition(
        name="get_wikipedia_evidence_pack",
        description=(
            "For ANY query: fetch Wikipedia summary + external references + related media "
            "(images and some videos via Wikimedia Commons). No API keys."
        ),
        parameters={
            "query": "string - any user query/topic",
            "max_references": "int - external links to return (default 8)",
            "max_page_media": "int - images sourced from page usage (default 6)",
            "max_commons_images": "int - Commons image search results (default 6)",
            "max_commons_videos": "int - Commons video search results (default 4)",
        },
    ),
    ToolDefinition(
        name="duckduckgo_instant_answer",
        description=(
            "DuckDuckGo Instant Answer (no key). Limited: not full search results; "
            "useful for abstracts + related-topic links."
        ),
        parameters={
            "query": "string - any user query/topic",
            "max_related": "int - related topics (default 10)",
        },
    ),
]


TOOL_HANDLERS: Dict[str, ToolHandler] = {
    "format_markdown_to_html": _tool_format_markdown,
    "get_datetime": _tool_get_datetime,
    "get_wikipedia_evidence_pack": _tool_wikipedia_evidence_pack,
    "duckduckgo_instant_answer": _tool_duckduckgo_instant_answer,
}


def list_tools() -> List[ToolDefinition]:
    """Return tool metadata for prompt-building and docs."""
    return TOOL_DEFINITIONS


def run_tool(action: str, parameters: Dict[str, Any] | None = None) -> str:
    """Dispatch to available tools by name."""
    handler = TOOL_HANDLERS.get(action)
    if not handler:
        return f"Unknown action: {action}"
    return handler(parameters or {})


def build_tools_prompt() -> str:
    """Human-readable tool spec used in gateway system prompts."""
    lines: List[str] = []
    for tool in TOOL_DEFINITIONS:
        params = ", ".join([f"{k} ({v})" for k, v in tool.parameters.items()])
        lines.append(f"- {tool.name}: {tool.description} Params: {params}")
    return "\n".join(lines)
