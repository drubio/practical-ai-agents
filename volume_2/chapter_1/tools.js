/**
 * Local deterministic tools for LLM agents.
 */

import * as chrono from "chrono-node";
import { randomUUID } from "crypto";

function normalizeWhitespace(text) {
  return String(text ?? "").replace(/\s+/g, " ").trim();
}

function escapeRegex(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function triggerMatch(promptLower, trigger) {
  if (trigger.includes(" ") || [":", "/", ".", "-"].some((ch) => trigger.includes(ch))) {
    return promptLower.includes(trigger);
  }
  return new RegExp(`\\b${escapeRegex(trigger)}\\b`).test(promptLower);
}

export function resolveDatetime(text) {
  const cleaned = normalizeWhitespace(text);
  if (!cleaned) {
    return { error: "No datetime text provided." };
  }

  const parsed = chrono.parseDate(cleaned);
  if (!parsed) {
    return { error: "Could not parse datetime." };
  }

  return {
    original: cleaned,
    resolved_iso: parsed.toISOString(),
    human_readable: parsed.toLocaleString("en-US", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit"
    })
  };
}

export function calculator(expression) {
  const cleaned = normalizeWhitespace(expression);
  if (!cleaned) {
    return { error: "No expression provided." };
  }

  if (!/^[0-9+\-*/().,\sA-Za-z_]+$/.test(cleaned)) {
    return { error: "Could not evaluate expression: invalid characters in expression." };
  }

  const scope = {
    pi: Math.PI,
    e: Math.E,
    sqrt: Math.sqrt,
    sin: Math.sin,
    cos: Math.cos,
    tan: Math.tan,
    log: Math.log,
    log10: Math.log10,
    exp: Math.exp,
    abs: Math.abs,
    ceil: Math.ceil,
    floor: Math.floor
  };

  try {
    const evaluator = new Function(...Object.keys(scope), `"use strict"; return (${cleaned});`);

    return {
      expression: cleaned,
      result: evaluator(...Object.values(scope))
    };
  } catch (error) {
    return {
      error: `Could not evaluate expression: ${error.message}`
    };
  }
}

export function generateUUID() {
  return {
    uuid: randomUUID()
  };
}


export const tools = {
  calculator,
  resolve_datetime: resolveDatetime,
  generate_uuid: generateUUID,
};

const TOOL_DESCRIPTIONS = {
  calculator: "Safely evaluate arithmetic expressions.",
  resolve_datetime: "Resolve date/time phrases into ISO and human-readable values.",
  generate_uuid: "Generate a unique UUID identifier.",
};

const DEFAULT_TOOL_PRIORITY = ["calculator", "resolve_datetime", "generate_uuid"];

const DEFAULT_KEYWORD_ROUTES = {
  calculator: ["calculate", "calc", "compute", "math", "equation", "percentage"],
  resolve_datetime: ["resolve datetime", "parse date", "parse datetime", "when is", "tomorrow", "next week", "next month", "today"],
  generate_uuid: ["uuid", "unique id", "ticket id", "identifier"],
};

export function routeToolForPrompt(prompt, availableToolNames = DEFAULT_TOOL_PRIORITY) {
  const text = String(prompt ?? "").trim();
  if (!text) return null;

  const available = new Set(availableToolNames);

  const calculatorMatch = text.match(/^(?:calculate|calc|compute)\s+(.+)$/i);
  if (calculatorMatch && available.has("calculator")) {
    return { name: "calculator", input: calculatorMatch[1].trim() };
  }

  if (["+", "-", "*", "/", "="].some((op) => text.includes(op)) && available.has("calculator")) {
    return { name: "calculator", input: text.replace(/=/g, " ").trim() };
  }

  const datetimeMatch = text.match(/^(?:resolve\s+datetime|parse\s+date(?:time)?|when\s+is)\s+(.+)$/i);
  if (datetimeMatch && available.has("resolve_datetime")) {
    return { name: "resolve_datetime", input: datetimeMatch[1].trim() };
  }

  const lower = text.toLowerCase();
  if (["tomorrow", "next week", "next month", "today", " at "].some((token) => lower.includes(token)) && available.has("resolve_datetime")) {
    return { name: "resolve_datetime", input: text };
  }

  const uuidMatch = text.match(
    /^(?:generate|create|make)\s+(?:a\s+)?(?:unique\s+)?(?:uuid|id|identifier|ticket id|ticket identifier)\b.*$/i
  );
  if (uuidMatch && available.has("generate_uuid")) {
    return { name: "generate_uuid", input: "" };
  }

  if (
    [
      "generate a unique id",
      "generate an id",
      "generate a uuid",
      "create a unique id",
      "create an id",
      "create a uuid",
      "new ticket id",
      "unique ticket id",
      "unique identifier"
    ].some((phrase) => lower.includes(phrase)) &&
    available.has("generate_uuid")
  ) {
    return { name: "generate_uuid", input: "" };
  }


  return null;
}

export function routeToolsForPrompt(prompt, availableToolNames = DEFAULT_TOOL_PRIORITY) {
  const text = String(prompt ?? "");
  const promptLower = text.toLowerCase();
  const available = new Set(availableToolNames);
  const selected = new Set();

  for (const [toolName, triggers] of Object.entries(DEFAULT_KEYWORD_ROUTES)) {
    if (!available.has(toolName)) continue;
    if (triggers.some((trigger) => triggerMatch(promptLower, trigger))) {
      selected.add(toolName);
    }
  }

  const hasMathExpression = ["+", "*", "/", "="].some((op) => text.includes(op)) || text.includes(" - ");
  if (hasMathExpression && available.has("calculator")) {
    selected.add("calculator");
  }

  if (!selected.size) {
    return DEFAULT_TOOL_PRIORITY.filter((name) => available.has(name));
  }

  return DEFAULT_TOOL_PRIORITY.filter((name) => available.has(name) && selected.has(name));
}

export function listTools() {
  return {
    tools: Object.keys(tools).map((name) => ({
      name,
      description: TOOL_DESCRIPTIONS[name]
    }))
  };
}

export function buildToolsPrompt() {
  const lines = ["You can use the following local deterministic tools:", ""];

  for (const [name, description] of Object.entries(TOOL_DESCRIPTIONS)) {
    lines.push(`- ${name}: ${description}`);
  }

  lines.push(
    "",
    "When a tool is needed, respond in this format:",
    "TOOL: <tool_name>",
    "INPUT: <tool_input>",
    "",
    "After receiving the tool result, continue your reasoning using the observation."
  );

  return lines.join("\n");
}

export function runTool(name, inputData) {
  const selectedTool = tools[name];

  if (!selectedTool) {
    return { error: `Tool '${name}' not found.` };
  }

  try {
    return selectedTool(inputData);
  } catch (error) {
    return {
      error: `Tool '${name}' failed: ${error.message}`
    };
  }
}
