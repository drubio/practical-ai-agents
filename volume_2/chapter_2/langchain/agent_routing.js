#!/usr/bin/env node

import * as z from "zod";
import { tool } from "langchain";

import * as tools from "../../chapter_1/tools.js";

export const CHAPTER_1_TOOL_NAMES = ["summarize_text"];

export const ALL_TOOL_NAMES = [
  "summarize_text",
  "extract_keywords",
  "extract_tasks",
  "score_priority",
  "route_workflow",
  "parse_content",
  "resolve_datetime",
  "format_json",
  "calculator",
  "analyze_text"
];

export function buildTools(logToolCall, logger) {
  return {
    summarize_text: tool(logToolCall(logger, "summarize_text", ({ text }) => tools.summarizeText(text)), {
      name: "summarize_text",
      description: "Summarize text.",
      schema: z.object({ text: z.string() })
    }),
    extract_keywords: tool(logToolCall(logger, "extract_keywords", ({ text }) => tools.extractKeywords(text)), {
      name: "extract_keywords",
      description: "Extract keywords.",
      schema: z.object({ text: z.string() })
    }),
    extract_tasks: tool(logToolCall(logger, "extract_tasks", ({ text }) => tools.extractTasks(text)), {
      name: "extract_tasks",
      description: "Extract tasks from text.",
      schema: z.object({ text: z.string() })
    }),
    score_priority: tool(logToolCall(logger, "score_priority", ({ text }) => tools.scorePriority(text)), {
      name: "score_priority",
      description: "Score priority from text.",
      schema: z.object({ text: z.string() })
    }),
    route_workflow: tool(logToolCall(logger, "route_workflow", ({ text }) => tools.routeWorkflow(text)), {
      name: "route_workflow",
      description: "Route workflow from text.",
      schema: z.object({ text: z.string() })
    }),
    parse_content: tool(logToolCall(logger, "parse_content", ({ content }) => tools.parseContent(content)), {
      name: "parse_content",
      description: "Parse content.",
      schema: z.object({ content: z.string() })
    }),
    resolve_datetime: tool(logToolCall(logger, "resolve_datetime", ({ text }) => tools.resolveDatetime(text)), {
      name: "resolve_datetime",
      description: "Resolve datetime from text.",
      schema: z.object({ text: z.string() })
    }),
    format_json: tool(logToolCall(logger, "format_json", ({ input }) => tools.formatJson(input)), {
      name: "format_json",
      description: "Format JSON-like input.",
      schema: z.object({ input: z.any() })
    }),
    calculator: tool(logToolCall(logger, "calculator", ({ expression }) => tools.calculator(expression)), {
      name: "calculator",
      description: "Evaluate expression.",
      schema: z.object({ expression: z.string() })
    }),
    analyze_text: tool(logToolCall(logger, "analyze_text", ({ text }) => tools.analyzeText(text)), {
      name: "analyze_text",
      description: "Analyze text.",
      schema: z.object({ text: z.string() })
    })
  };
}

export function selectTools(logToolCall, logger, toolNames) {
  const available = buildTools(logToolCall, logger);
  return toolNames.map((name) => available[name]);
}
