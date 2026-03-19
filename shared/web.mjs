import express from "express";
import cors from "cors";

function extractPythonStyleContent(payload) {
  const marker = "content=";
  const start = payload.indexOf(marker);
  if (start < 0) return null;
  const quoteIndex = start + marker.length;
  const quote = payload[quoteIndex];
  if (quote !== '"' && quote !== "'") return null;
  let i = quoteIndex + 1;
  let escaped = false;
  let value = "";
  while (i < payload.length) {
    const char = payload[i];
    if (escaped) {
      value += char === "n" ? "\n" : char === "r" ? "\r" : char === "t" ? "\t" : char;
      escaped = false;
      i += 1;
      continue;
    }
    if (char === "\\") {
      escaped = true;
      i += 1;
      continue;
    }
    if (char === quote) {
      return payload.slice(i + 1).includes("additional_kwargs=") ? value : null;
    }
    value += char;
    i += 1;
  }
  return null;
}

export function normalizeResponseText(payload) {
  if (payload == null) return "";
  if (typeof payload === "string") {
    const extracted = extractPythonStyleContent(payload);
    if (typeof extracted === "string") return extracted;
    try {
      const maybeJson = JSON.parse(payload);
      if (maybeJson && typeof maybeJson === "object") {
        for (const key of ["answer", "distilled", "content", "text", "message", "summary", "response"]) {
          const value = maybeJson[key];
          if (typeof value === "string" && value.trim()) return value;
        }
      }
    } catch {}
    return payload;
  }
  if (typeof payload === "object") {
    for (const key of ["content", "text", "message", "answer", "final_answer", "distilled", "summary", "response"]) {
      const value = payload[key];
      if (typeof value === "string" && value.trim()) return value;
    }
    return JSON.stringify(payload);
  }
  return String(payload);
}

export function chunkText(text, chunkSize = 28) {
  const clean = text || "";
  if (!clean) return [""];
  const chunks = [];
  for (let index = 0; index < clean.length; index += chunkSize) {
    chunks.push(clean.slice(index, index + chunkSize));
  }
  return chunks;
}

export async function* iterTextChunks(text, chunkSize = 28, delayMs = 0) {
  for (const part of chunkText(text, chunkSize)) {
    if (delayMs > 0) await new Promise((resolve) => setTimeout(resolve, delayMs));
    yield part;
  }
}

export function toSseLine(data) {
  return `data: ${JSON.stringify(data)}\n\n`;
}

export async function captureConsoleOutputAsync(fn) {
  const originalLog = console.log;
  const logs = [];
  console.log = (...args) => logs.push(args.join(" "));
  try {
    const result = await fn();
    return { result, logs: logs.join("\n") };
  } finally {
    console.log = originalLog;
  }
}

export function buildManager(managerClassOrFactory) {
  if (typeof managerClassOrFactory !== "function") {
    throw new Error("Invalid manager class/factory provided");
  }
  try {
    return new managerClassOrFactory();
  } catch {
    return managerClassOrFactory();
  }
}

export function supportsMemory(manager) {
  return Boolean(manager?.memoryEnabled && typeof manager.getHistory === "function" && typeof manager.resetMemory === "function");
}

export function supportsMemoryRetrieval(manager) {
  return Boolean(manager?.retrievalMemoryEnabled && typeof manager.getHistory === "function" && typeof manager.resetMemory === "function");
}

export function supportsSessionMemory(manager) {
  return supportsMemory(manager) || supportsMemoryRetrieval(manager);
}

export function supportsCoagent(manager) {
  return Boolean(manager?.coagent);
}

export function supportsHistory(manager) {
  return typeof manager?.getHistory === "function";
}

export function supportsResetMemory(manager) {
  return typeof manager?.resetMemory === "function";
}

export function createExpressApp() {
  const app = express();
  app.use(cors());
  app.use(express.json());
  return app;
}

export function resolveSessionId(...candidates) {
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim()) return candidate;
  }
  return "default";
}

export function resultIsSuccess(result) {
  return Boolean(result && typeof result === "object" && result.success);
}

export async function* streamTextSse(text, chunkSize = 28, delayMs = 0, eventType = "chunk") {
  for await (const part of iterTextChunks(text, chunkSize, delayMs)) {
    if (part) yield toSseLine({ type: eventType, content: part });
  }
}

