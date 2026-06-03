import { existsSync, readFileSync, writeFileSync } from "fs";
import readline from "node:readline";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatOpenAI } from "@langchain/openai";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

import {
  ALL_MODEL_IDENTIFIERS,
  getAllProviders,
  getApiKey,
  getDisplayName,
  getIdentifierMappings,
  getModelConfig,
  resolveModelConfig,
  sortProvidersByDisplayOrder,
} from "./llm_models.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const sharedEnvPath = join(__dirname, ".env");

if (existsSync(sharedEnvPath)) {
  for (const rawLine of readFileSync(sharedEnvPath, "utf-8").split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) continue;
    const [rawKey, ...rest] = line.split("=");
    const key = rawKey.trim();
    const value = rest.join("=").trim().replace(/^['"]|['"]$/g, "");
    if (key && !(key in process.env)) process.env[key] = value;
  }
}

export function normalizeResponseText(payload) {
  if (payload == null) return "";
  if (typeof payload === "string") {
    const parts = payload.match(/content=(['"])((?:\\.|(?!\1).)*)\1\s+additional_kwargs=/s);
    if (parts) {
      try {
        return JSON.parse(parts[1] === '"' ? `"${parts[2]}"` : JSON.stringify(parts[2]));
      } catch {
        return parts[2];
      }
    }
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

export { getAllProviders, getApiKey, getDisplayName, sortProvidersByDisplayOrder };
export function getSelectedModelDetails(selectedModel) {
  const config = resolveModelConfig(selectedModel);
  return {
    provider: config.provider,
    displayName: getDisplayName(config.provider),
    selectedModel: config.model,
    selectedModelIdentifier: config.name,
    selectedModelTier: config.tier,
  };
}

export function createLangChainModel(selectedModel, { temperature = 0.7, maxTokens = 1000 } = {}) {
  const config = resolveModelConfig(selectedModel);
  const provider = config.provider;
  const modelName = config.model;
  if (provider === "anthropic") {
    return new ChatAnthropic({
      apiKey: getApiKey(provider),
      model: modelName,
      temperature,
      maxTokens,
    });
  }
  if (provider === "openai") {
    return new ChatOpenAI({
      apiKey: getApiKey(provider),
      model: modelName,
      temperature,
      maxTokens,
    });
  }
  if (provider === "google") {
    return new ChatGoogleGenerativeAI({
      apiKey: getApiKey(provider),
      model: modelName,
      temperature,
      maxTokens,
    });
  }
  if (provider === "xai") {
    return new ChatOpenAI({
      apiKey: getApiKey(provider),
      configuration: { baseURL: process.env.XAI_API_BASE || "https://api.x.ai/v1" },
      model: modelName,
      temperature,
      maxTokens,
    });
  }
  if (provider === "deepseek") {
    return new ChatOpenAI({
      apiKey: getApiKey(provider),
      configuration: { baseURL: process.env.DEEPSEEK_API_BASE || "https://api.deepseek.com" },
      model: modelName,
      temperature,
      maxTokens,
    });
  }
  throw new Error(`Unsupported provider: ${provider}`);
}
let sharedRl = null;
export function getSharedAsk() {
  if (!sharedRl) sharedRl = readline.createInterface({ input: process.stdin, output: process.stdout });
  return (prompt) => new Promise((resolve) => sharedRl.question(prompt, resolve));
}
export function closeSharedAsk() { if (sharedRl) { sharedRl.close(); sharedRl = null; } }

export function parseStructuredJsonResponse(raw) {
  let content = "";
  if (raw == null) content = "";
  else if (typeof raw === "string") content = raw.trim();
  else if (typeof raw === "object" && typeof raw.content === "string") content = raw.content.trim();
  else if (typeof raw === "object" && !Array.isArray(raw)) content = JSON.stringify(raw);
  else content = normalizeResponseText(raw).trim();
  if (!content) throw new Error("Structured content is empty");
  const parts = content.match(/content=(['"])((?:\\.|(?!\1).)*)\1\s+additional_kwargs=/s);
  if (parts) {
    try { content = JSON.parse(parts[1] === '"' ? `"${parts[2]}"` : JSON.stringify(parts[2])).trim(); } catch {}
  }
  if (content.startsWith("```json")) content = content.slice(7);
  if (content.startsWith("```")) content = content.slice(3);
  if (content.endsWith("```")) content = content.slice(0, -3);
  content = content.trim();
  try {
    const parsed = JSON.parse(content);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
  } catch {}
  if (content.startsWith("[") && content.endsWith("]")) {
    try {
      const blocks = Function(`"use strict"; return (${content});`)();
      if (Array.isArray(blocks)) {
        for (const block of blocks) {
          const maybeText = block?.text || block?.content;
          if (typeof maybeText === "string" && maybeText.trim()) return parseStructuredJsonResponse(maybeText);
        }
      }
    } catch {}
  }
  const start = content.indexOf("{");
  if (start === -1) throw new Error("No JSON object found in structured content");
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = start; i < content.length; i += 1) {
    const ch = content[i];
    if (inString) {
      if (escape) escape = false;
      else if (ch === "\\") escape = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') { inString = true; continue; }
    if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        const parsed = JSON.parse(content.slice(start, i + 1));
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
      }
    }
  }
  throw new Error("Parsed structured content is not a JSON object");
}

export function buildTaskPrompt(topic) {
  const text = String(topic ?? "").trim();
  if (!text) return "";
  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (lines.length <= 1) return text;
  const checklist = lines.map((line, index) => `${index + 1}. ${line}`).join("\n");
  return `${text}\n\nTask checklist (every item is required, including the final line):\n${checklist}\n\nDo not skip any checklist item.`;
}

export function getChapterLogger(name) {
  const levels = { debug: 10, info: 20, warn: 30, error: 40 };
  const min = levels[(process.env.LOG_LEVEL || "info").toLowerCase()] ?? levels.info;
  const log = (level, message, ...args) => {
    if ((levels[level] ?? 100) < min) return;
    console.log(`${new Date().toISOString()} | ${level.toUpperCase()} | ${name} | ${message}`, ...args);
  };
  return { debug: (m, ...a) => log("debug", m, ...a), info: (m, ...a) => log("info", m, ...a), warn: (m, ...a) => log("warn", m, ...a), error: (m, ...a) => log("error", m, ...a) };
}

export function logToolCall(logger, toolName, fn) {
  return (arg = undefined) => {
    logger.info(`Tool call | name=${toolName} | input=%o`, arg);
    const result = fn(arg);
    logger.info(`Tool result | name=${toolName} | output=%o`, result);
    return result;
  };
}

export async function getUserParameters(ask) {
  const tempInput = await ask("Temperature (0.0-2.0, default 0.7): ");
  let temperature = 0.7;
  if (tempInput.trim()) {
    const parsed = parseFloat(tempInput);
    temperature = !Number.isNaN(parsed) ? Math.max(0.0, Math.min(2.0, parsed)) : 0.7;
  }
  const tokensInput = await ask("Max tokens (default 1000): ");
  let maxTokens = 1000;
  if (tokensInput.trim()) {
    const parsed = parseInt(tokensInput, 10);
    maxTokens = !Number.isNaN(parsed) ? Math.max(1, Math.min(4000, parsed)) : 1000;
  }
  return { temperature, maxTokens };
}

export function displayProviderResponse(provider, response, framework = "") {
  const providerDisplay = `${getDisplayName(provider)}${framework ? ` (${framework})` : ""} answered:`;
  console.log(`\n=== ${providerDisplay} ===`);
  const configParts = [];
  if (typeof response.temperature !== "undefined") configParts.push(`temp: ${response.temperature}`);
  if (typeof response.maxTokens !== "undefined") configParts.push(`max_tokens: ${response.maxTokens}`);
  else if (typeof response.max_tokens !== "undefined") configParts.push(`max_tokens: ${response.max_tokens}`);
  if (response.model) configParts.push(`model: ${response.model}`);
  if (configParts.length > 0) console.log(`[${configParts.join(", ")}]`);
  if (response.success) {
    const raw = response.response;
    if (raw && typeof raw === "object") console.log(JSON.stringify(raw, null, 2));
    else console.log(normalizeResponseText(raw) || "No response");
  } else {
    console.log(`Error: ${response.error || "Unknown error"}`);
  }
  console.log("=".repeat(60));
}

export async function getUserChoice(options, prompt, ask) {
  console.log(`\n${prompt}`);
  options.forEach((option, i) => console.log(`${i + 1}. ${option}`));
  while (true) {
    const answer = (await ask(`Select an option (1-${options.length}, default 1): `)).trim();
    const choice = (answer === "" ? 1 : parseInt(answer, 10)) - 1;
    if (choice >= 0 && choice < options.length) return choice;
    console.log("Invalid selection. Please try again.");
  }
}

export const MODEL_PROVIDER_PREFIXES = [
  ["google_genai_", "Google"],
  ["anthropic_", "Anthropic"],
  ["openai_", "OpenAI"],
  ["xai_", "xAI"],
  ["deepseek_", "DeepSeek"],
];

export function providerAndModelName(modelIdentifier) {
  for (const [prefix, providerName] of MODEL_PROVIDER_PREFIXES) {
    if (modelIdentifier.startsWith(prefix)) return [providerName, modelIdentifier.slice(prefix.length)];
  }
  const [providerName, ...modelParts] = modelIdentifier.split("_");
  return [providerName ? providerName.charAt(0).toUpperCase() + providerName.slice(1) : "Other", modelParts.join("_") || modelIdentifier];
}

export function compactModelSelectionLines(modelIdentifiers) {
  const lines = [];
  let currentProvider = null;
  let currentOptions = [];
  const flushCurrent = () => {
    if (currentProvider && currentOptions.length > 0) lines.push(`${currentProvider}: ${currentOptions.join(" | ")}`);
    currentOptions = [];
  };
  modelIdentifiers.forEach((modelIdentifier, index) => {
    const [providerName, modelName] = providerAndModelName(modelIdentifier);
    if (providerName !== currentProvider || currentOptions.length === 3) {
      flushCurrent();
      currentProvider = providerName;
    }
    currentOptions.push(`${index + 1}. ${modelName}${index === 0 ? " [first]" : ""}`);
  });
  flushCurrent();
  return lines;
}

export function modelIdentifiersForProviders(providers = null) {
  const providerSet = providers ? new Set(providers) : null;
  const mappings = getIdentifierMappings();
  return ALL_MODEL_IDENTIFIERS.filter((identifier) => mappings[identifier] && (!providerSet || providerSet.has(mappings[identifier].provider)));
}

export function modelOptionLabel(modelIdentifier) {
  const config = getIdentifierMappings()[modelIdentifier];
  return `${config.model} (${config.tier}; ${modelIdentifier})`;
}

export async function selectModelIdentifier(modelIdentifiers, ask, prompt = "Select a model:") {
  const choiceIdx = await getUserChoice(modelIdentifiers.map((identifier) => modelOptionLabel(identifier)), prompt, ask);
  return modelIdentifiers[choiceIdx];
}

export async function selectProviderModelIdentifier(providers, ask) {
  const providerIdx = await getUserChoice(
    providers.map((provider) => `${getDisplayName(provider)} (${modelIdentifiersForProviders([provider]).length} models)`),
    "Select a provider:",
    ask,
  );
  const provider = providers[providerIdx];
  return selectModelIdentifier(modelIdentifiersForProviders([provider]), ask, `Select a ${getDisplayName(provider)} model:`);
}
