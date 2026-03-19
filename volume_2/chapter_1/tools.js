import { randomUUID } from "node:crypto";

function normalizeWhitespace(text) {
  return String(text || "").replace(/\s+/g, " ").trim();
}

function sanitizeMathCandidate(candidate) {
  let sanitized = normalizeWhitespace(candidate).replace(/[.,;:]+$/, "");

  while ((sanitized.match(/\)/g) || []).length > (sanitized.match(/\(/g) || []).length && sanitized.endsWith(")")) {
    sanitized = sanitized.slice(0, -1).trimEnd();
  }

  while ((sanitized.match(/\(/g) || []).length > (sanitized.match(/\)/g) || []).length && sanitized.startsWith("(")) {
    sanitized = sanitized.slice(1).trimStart();
  }

  return sanitized;
}

function extractMathExpression(text) {
  const cleaned = normalizeWhitespace(text);
  if (!cleaned) return "";

  const parenthesized = [...cleaned.matchAll(/\(([^()]+)\)/g)];
  for (let i = parenthesized.length - 1; i >= 0; i -= 1) {
    const candidate = sanitizeMathCandidate(parenthesized[i][1]);
    if (/\d/.test(candidate) && /[+\-*/]/.test(candidate)) return candidate;
  }

  for (const match of cleaned.matchAll(/[\d\s+\-*/().,]+/g)) {
    const candidate = sanitizeMathCandidate(match[0]);
    if (/\d/.test(candidate) && /[+\-*/]/.test(candidate)) return candidate;
  }

  return cleaned;
}

function applyTimeKeywords(text, date) {
  const lowered = String(text || "").toLowerCase();
  const adjusted = new Date(date);

  if (/\bnoon\b/.test(lowered)) {
    adjusted.setHours(12, 0, 0, 0);
  } else if (/\bmidnight\b/.test(lowered)) {
    adjusted.setHours(0, 0, 0, 0);
  }

  return adjusted;
}

function parseRelativeDate(text) {
  const normalized = normalizeWhitespace(text).toLowerCase();
  if (!normalized) return null;

  const now = new Date();
  const result = new Date(now);
  result.setSeconds(0, 0);

  const dayOffset =
    /\bday after tomorrow\b/.test(normalized) ? 2
      : /\btomorrow\b/.test(normalized) ? 1
        : /\byesterday\b/.test(normalized) ? -1
          : /\btoday\b/.test(normalized) ? 0
            : null;

  if (dayOffset == null) return null;
  result.setDate(result.getDate() + dayOffset);

  const meridiemMatch = normalized.match(/\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b/);
  const twentyFourHourMatch = normalized.match(/\b(?:at\s+)?(\d{1,2}):(\d{2})\b/);

  if (/\bnoon\b/.test(normalized)) {
    result.setHours(12, 0, 0, 0);
  } else if (/\bmidnight\b/.test(normalized)) {
    result.setHours(0, 0, 0, 0);
  } else if (meridiemMatch) {
    let hour = Number(meridiemMatch[1]) % 12;
    const minute = Number(meridiemMatch[2] || "0");
    if (meridiemMatch[3] === "pm") hour += 12;
    result.setHours(hour, minute, 0, 0);
  } else if (twentyFourHourMatch) {
    result.setHours(Number(twentyFourHourMatch[1]), Number(twentyFourHourMatch[2]), 0, 0);
  }

  return result;
}

function safeEval(node) {
  if (node.type === "Literal" && typeof node.value === "number") return node.value;

  if (node.type === "UnaryExpression" && ["+", "-"].includes(node.operator)) {
    const value = safeEval(node.argument);
    return node.operator === "-" ? -value : value;
  }

  if (node.type === "BinaryExpression") {
    const left = safeEval(node.left);
    const right = safeEval(node.right);
    switch (node.operator) {
      case "+": return left + right;
      case "-": return left - right;
      case "*": return left * right;
      case "/": return left / right;
      case "%": return left % right;
      case "**": return left ** right;
      default:
        throw new Error(`Unsupported operator: ${node.operator}`);
    }
  }

  throw new Error(`Unsupported expression: ${node.type}`);
}

export function calculator(expression) {
  const candidate = extractMathExpression(expression);
  if (!candidate) return { error: "No expression provided." };

  if (!/^[\d\s+\-*/%.()]+$/.test(candidate)) {
    return { error: "Could not evaluate expression: expression contains unsupported characters." };
  }

  try {
    const ast = Function(`"use strict"; return (${candidate.replace(/\^/g, "**")});`)();
    if (typeof ast !== "number" || Number.isNaN(ast)) {
      return { error: "Could not evaluate expression: result is not a valid number." };
    }
    return { expression: candidate, result: ast };
  } catch (error) {
    return { error: `Could not evaluate expression: ${error.message}` };
  }
}

export function resolveDatetime(text) {
  const normalized = normalizeWhitespace(text);
  if (!normalized) return { error: "No datetime text provided." };

  let date = parseRelativeDate(normalized);
  if (!date) {
    const parsed = new Date(normalized);
    if (Number.isNaN(parsed.getTime())) {
      return { error: "Could not parse datetime: unsupported date format." };
    }
    date = applyTimeKeywords(normalized, parsed);
  }

  return {
    original: normalized,
    resolved_iso: date.toISOString(),
    human_readable: new Intl.DateTimeFormat("en-US", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: true,
      timeZone: "UTC"
    }).format(date)
  };
}

export function generateUUID() {
  return { uuid: randomUUID() };
}
