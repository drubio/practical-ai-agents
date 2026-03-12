#!/usr/bin/env node

import {
  analyzeText,
  calculator,
  extractKeywords,
  extractTasks,
  formatJson,
  parseContent,
  resolveDatetime,
  routeWorkflow,
  scorePriority,
  summarizeText
} from "../../chapter_1/tools.js";

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

export function buildToolMap(logToolCall, logger) {
  return {
    summarize_text: logToolCall(logger, "summarize_text", summarizeText),
    extract_keywords: logToolCall(logger, "extract_keywords", extractKeywords),
    extract_tasks: logToolCall(logger, "extract_tasks", extractTasks),
    score_priority: logToolCall(logger, "score_priority", scorePriority),
    route_workflow: logToolCall(logger, "route_workflow", routeWorkflow),
    parse_content: logToolCall(logger, "parse_content", parseContent),
    resolve_datetime: logToolCall(logger, "resolve_datetime", resolveDatetime),
    format_json: logToolCall(logger, "format_json", formatJson),
    calculator: logToolCall(logger, "calculator", calculator),
    analyze_text: logToolCall(logger, "analyze_text", analyzeText)
  };
}

export function selectToolMap(logToolCall, logger, toolNames) {
  const available = buildToolMap(logToolCall, logger);
  return Object.fromEntries(toolNames.map((name) => [name, available[name]]));
}
