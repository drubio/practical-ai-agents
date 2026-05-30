"""Chapter 7 tool utilities (Python)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List
import json
import re
import urllib.parse
import urllib.request
import urllib.error
import time


@dataclass(frozen=True)
class ToolDefinition:
    """Serializable metadata that can be shared with prompt templates."""

    name: str
    description: str
    parameters: Dict[str, str]


ToolHandler = Callable[[Dict[str, Any]], str]

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_REST = "https://en.wikipedia.org/api/rest_v1"
_COMMONS_API = "https://commons.wikimedia.org/w/api.php"


def _wiki_get_json(base_url: str, params: Dict[str, Any] | None = None, attempts: int = 3) -> Dict[str, Any]:
    encoded = urllib.parse.urlencode(params or {})
    url = f"{base_url}?{encoded}" if encoded else base_url

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers={"User-Agent": "software-with-llms-ch7/1.0"}, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code in (429, 500, 502, 503, 504) and attempt < attempts:
                time.sleep(0.25 * attempt)
                continue
            raise
        except Exception as error:
            last_error = error
            if attempt < attempts:
                time.sleep(0.25 * attempt)
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch {url}")


def _commons_resolve_file_urls(file_titles: List[str]) -> List[str]:
    if not file_titles:
        return []

    data = _wiki_get_json(
        _COMMONS_API,
        {
            "action": "query",
            "titles": "|".join(file_titles[:20]),
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "format": "json",
            "origin": "*",
        },
    )

    urls: List[str] = []
    for page in (data.get("query", {}).get("pages", {}) or {}).values():
        for info in page.get("imageinfo", []) or []:
            candidate = info.get("url")
            if isinstance(candidate, str) and candidate.startswith("http") and candidate not in urls:
                urls.append(candidate)
    return urls


def get_wikipedia_evidence_pack(
    query: str,
    max_references: int = 8,
    max_page_media: int = 6,
    max_commons_images: int = 6,
    max_commons_videos: int = 4,
) -> Dict[str, Any]:
    """Fetch Wikipedia summary, references, and media in one consolidated flow."""
    query = (query or "").strip()
    if not query:
        return {"error": "Query is empty."}

    search_data = _wiki_get_json(
        _WIKI_API,
        {
            "action": "opensearch",
            "search": query,
            "limit": 1,
            "namespace": 0,
            "format": "json",
            "origin": "*",
        },
    )
    titles = search_data[1] if len(search_data) > 1 else []
    page_urls = search_data[3] if len(search_data) > 3 else []
    if not titles:
        return {"error": "No Wikipedia results found.", "query": query}

    title = str(titles[0])
    fallback_page_url = str(page_urls[0]) if page_urls else ""

    summary_data = _wiki_get_json(
        f"{_WIKI_REST}/page/summary/{urllib.parse.quote(title.replace(' ', '_'))}",
    )
    summary = str(summary_data.get("extract") or "")
    wiki_url = summary_data.get("content_urls", {}).get("desktop", {}).get("page") or fallback_page_url

    images: List[str] = []
    for candidate in [
        (summary_data.get("originalimage", {}) or {}).get("source"),
        (summary_data.get("thumbnail", {}) or {}).get("source"),
    ]:
        if isinstance(candidate, str) and candidate.startswith("http") and candidate not in images:
            images.append(candidate)

    page_images_data = _wiki_get_json(
        _WIKI_API,
        {
            "action": "query",
            "titles": title,
            "prop": "images",
            "imlimit": int(max_page_media),
            "format": "json",
            "origin": "*",
        },
    )
    page_file_titles: List[str] = []
    for page in (page_images_data.get("query", {}).get("pages", {}) or {}).values():
        for img in page.get("images", []) or []:
            name = img.get("title")
            if isinstance(name, str) and name.startswith("File:") and re.search(r"\.(png|jpg|jpeg|gif|svg|webm|ogv)$", name, re.I):
                page_file_titles.append(name)
            if len(page_file_titles) >= max_page_media:
                break

    for url in _commons_resolve_file_urls(page_file_titles):
        if url not in images:
            images.append(url)

    references_data = _wiki_get_json(
        _WIKI_API,
        {
            "action": "parse",
            "page": title,
            "prop": "externallinks",
            "format": "json",
            "origin": "*",
        },
    )
    references: List[str] = []
    for link in references_data.get("parse", {}).get("externallinks", []) or []:
        if isinstance(link, str) and link.startswith("http"):
            references.append(link)
        if len(references) >= int(max_references):
            break

    def commons_media(kind: str, limit: int) -> List[Dict[str, str]]:
        search_query = f"{query} filetype:video" if kind == "video" else f"{query} filetype:bitmap"
        commons_search = _wiki_get_json(
            _COMMONS_API,
            {
                "action": "query",
                "list": "search",
                "srsearch": search_query,
                "srlimit": int(limit),
                "format": "json",
                "origin": "*",
            },
        )

        file_titles: List[str] = []
        for result in commons_search.get("query", {}).get("search", []) or []:
            name = result.get("title")
            if not isinstance(name, str) or not name.startswith("File:"):
                continue
            if kind == "video" and not re.search(r"\.(webm|ogv)$", name, re.I):
                continue
            if kind == "image" and not re.search(r"\.(png|jpg|jpeg|gif|svg)$", name, re.I):
                continue
            file_titles.append(name)

        urls = _commons_resolve_file_urls(file_titles)
        return [
            {
                "file_title": file_title,
                "file_page": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(file_title.replace(' ', '_'))}",
                "url": url,
            }
            for file_title, url in zip(file_titles, urls)
        ]

    return {
        "query": query,
        "topic": title,
        "page_url": wiki_url,
        "summary": summary,
        "references": references,
        "media": {
            "images": images[: int(max_page_media) + 2],
            "commons_images": commons_media("image", int(max_commons_images)),
            "commons_videos": commons_media("video", int(max_commons_videos)),
        },
    }


def _tool_wikipedia_evidence_pack(parameters: Dict[str, Any]) -> str:
    pack = get_wikipedia_evidence_pack(
        query=str(parameters.get("query", "")).strip(),
        max_references=int(parameters.get("max_references", 8)),
        max_page_media=int(parameters.get("max_page_media", 6)),
        max_commons_images=int(parameters.get("max_commons_images", 6)),
        max_commons_videos=int(parameters.get("max_commons_videos", 4)),
    )
    return json.dumps(pack, ensure_ascii=False, indent=2)


TOOL_DEFINITIONS: List[ToolDefinition] = [
    ToolDefinition(
        name="get_wikipedia_evidence_pack",
        description="For any query: fetch Wikipedia summary + references + media from Wikimedia Commons.",
        parameters={
            "query": "string - any user query/topic",
            "max_references": "int - external links to return (default 8)",
            "max_page_media": "int - images sourced from page usage (default 6)",
            "max_commons_images": "int - Commons image search results (default 6)",
            "max_commons_videos": "int - Commons video search results (default 4)",
        },
    ),
]


TOOL_HANDLERS: Dict[str, ToolHandler] = {
    "get_wikipedia_evidence_pack": _tool_wikipedia_evidence_pack,
}


def list_tools() -> List[ToolDefinition]:
    return TOOL_DEFINITIONS


def run_tool(action: str, parameters: Dict[str, Any] | None = None) -> str:
    handler = TOOL_HANDLERS.get(action)
    if not handler:
        return f"Unknown action: {action}"

    params = parameters or {}
    try:
        return handler(params)
    except Exception as error:
        return json.dumps({"error": str(error), "query": str(params.get("query", "")).strip()}, ensure_ascii=False, indent=2)


def build_tools_prompt() -> str:
    lines: List[str] = []
    for tool in TOOL_DEFINITIONS:
        params = ", ".join([f"{k} ({v})" for k, v in tool.parameters.items()])
        lines.append(f"- {tool.name}: {tool.description} Params: {params}")
    return "\n".join(lines)
