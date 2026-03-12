#!/usr/bin/env node

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

export function buildToolMap(logToolCall, logger) {
  return {
    summarize_text: logToolCall(logger, "summarize_text", tools.summarizeText),
    extract_keywords: logToolCall(logger, "extract_keywords", tools.extractKeywords),
    extract_tasks: logToolCall(logger, "extract_tasks", tools.extractTasks),
    score_priority: logToolCall(logger, "score_priority", tools.scorePriority),
    route_workflow: logToolCall(logger, "route_workflow", tools.routeWorkflow),
    parse_content: logToolCall(logger, "parse_content", tools.parseContent),
    resolve_datetime: logToolCall(logger, "resolve_datetime", tools.resolveDatetime),
    format_json: logToolCall(logger, "format_json", tools.formatJson),
    calculator: logToolCall(logger, "calculator", tools.calculator),
    analyze_text: logToolCall(logger, "analyze_text", tools.analyzeText)
  };
}

export function selectToolMap(logToolCall, logger, toolNames) {
  const available = buildToolMap(logToolCall, logger);
  return Object.fromEntries(toolNames.map((name) => [name, available[name]]));
}
