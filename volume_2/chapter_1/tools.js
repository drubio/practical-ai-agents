/**
 * Local deterministic tools for LLM agents.
 */

import * as chrono from "chrono-node";
import { randomUUID } from "crypto";

function normalizeWhitespace(text) {
  return String(text ?? "").replace(/\s+/g, " ").trim();
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
    const evaluator = new Function(
      ...Object.keys(scope),
      `"use strict"; return (${cleaned});`
    );

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
  generate_uuid: generateUUID
};

const TOOL_DESCRIPTIONS = {
  calculator: "Safely evaluate arithmetic expressions.",
  resolve_datetime: "Resolve date/time phrases into ISO and human-readable values.",
  generate_uuid: "Generate a unique UUID identifier."
};

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
  const tool = tools[name];

  if (!tool) {
    return { error: `Tool '${name}' not found.` };
  }

  try {
    return tool(inputData);
  } catch (error) {
    return {
      error: `Tool '${name}' failed: ${error.message}`
    };
  }
}
