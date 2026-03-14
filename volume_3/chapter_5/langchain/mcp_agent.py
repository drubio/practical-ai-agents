#!/usr/bin/env python3
"""LangChain agent that consumes summarize_text through an MCP server."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
CHAPTER_1_ROOT = REPO_ROOT / "volume_2" / "chapter_1"
if str(CHAPTER_1_ROOT) not in sys.path:
    sys.path.append(str(CHAPTER_1_ROOT))

from utils import (  # type: ignore[import-not-found]
    build_common_parser,
    extract_output_text,
    get_chapter_logger,
    run_mode,
    select_startup_model,
)

from langchain.agents import create_agent

import tools  # type: ignore[import-not-found]
from models import ALL_MODEL_IDENTIFIERS, get_identifier_mappings  # type: ignore[import-not-found]


logger = get_chapter_logger("volume_3.chapter_5.langchain.mcp_agent")


def run_summarize_text_mcp_server() -> None:
    """Start a local MCP server that exposes summarize_text over stdio."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("chapter_1_tools")

    @server.tool()
    def summarize_text(text: str, max_sentences: int = 3) -> Dict[str, Any]:
        """Summarize text with a deterministic extractor."""
        return tools.summarize_text(text=text, max_sentences=max_sentences)

    server.run(transport="stdio")


class LangChainMCPAgentManager:
    framework = "LangChain MCP Agent"
    tool_names = ["summarize_text (via MCP server)"]
    model_identifiers = ALL_MODEL_IDENTIFIERS
    tool_trigger_help = (
        "Tools are selected automatically from your prompt; you do not need to type a tool name. "
        "If you want a specific behavior, ask explicitly (for example: 'summarize this')."
    )

    def __init__(self, model: str):
        config = get_identifier_mappings().get(model)
        self.provider = config.provider if config else "unknown"
        self.model = config.model if config else model
        self._mcp_client = None
        logger.info("Initializing LangChain MCP agent | provider=%s | model=%s", self.provider, self.model)
        self.agent = asyncio.run(self._build_agent())

    async def _build_agent(self):
        from langchain_mcp_adapters.client import MultiServerMCPClient

        self._mcp_client = MultiServerMCPClient(
            {
                "chapter_1_tools": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [str(Path(__file__).resolve()), "--run-mcp-server"],
                }
            }
        )

        mcp_tools = await self._mcp_client.get_tools()
        return create_agent(
            model=f"{self.provider}:{self.model}",
            tools=mcp_tools,
            system_prompt=(
                "You are an AI assistant that can use MCP tools. "
                "When summarization is needed, call summarize_text from the MCP server. "
                "Think step-by-step, use tools when needed, and return a concise final answer."
            ),
        )

    def ask_question(self, topic: str) -> Dict[str, Any]:
        try:
            logger.info("Processing prompt | chars=%s", len(topic))
            result = asyncio.run(self.agent.ainvoke({"messages": [{"role": "user", "content": topic}]}))
            return {
                "success": True,
                "provider": self.provider,
                "model": self.model,
                "prompt": topic,
                "response": extract_output_text(result),
            }
        except Exception as exc:
            logger.exception("LangChain MCP ask_question failed")
            return {
                "success": False,
                "provider": self.provider,
                "model": self.model,
                "prompt": topic,
                "error": str(exc),
                "response": None,
            }


def _build_parser() -> argparse.ArgumentParser:
    parser = build_common_parser("LangChain MCP Agent")
    parser.add_argument("--run-mcp-server", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.run_mcp_server:
        run_summarize_text_mcp_server()
        return

    startup_model = select_startup_model(ALL_MODEL_IDENTIFIERS, args.mode, args.model_identifier)
    manager = LangChainMCPAgentManager(model=startup_model)
    run_mode(manager, args.mode, args.host, args.port, args.stream)


if __name__ == "__main__":
    main()
