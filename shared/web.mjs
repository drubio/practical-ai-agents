import express from "express";
import cors from "cors";
import {
  ALL_MODEL_IDENTIFIERS,
  getAllProviders,
  getDisplayName,
  getIdentifierMappings,
  sortProvidersByDisplayOrder,
} from "./llm_models.mjs";

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
  const originalWrite = process.stdout.write.bind(process.stdout);
  const logs = [];
  console.log = (...args) => logs.push(args.join(" "));
  process.stdout.write = (chunk, encoding, callback) => {
    const writeCallback = typeof encoding === "function" ? encoding : callback;
    const writeEncoding = typeof encoding === "string" ? encoding : "utf8";
    logs.push(Buffer.isBuffer(chunk) ? chunk.toString(writeEncoding) : String(chunk));
    if (typeof writeCallback === "function") writeCallback();
    return true;
  };
  try {
    const result = await fn();
    return { result, logs: logs.join("") };
  } finally {
    console.log = originalLog;
    process.stdout.write = originalWrite;
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

export function logApiRequest(req, res) {
  const startedAt = Date.now();
  res.on("finish", () => {
    const durationMs = Date.now() - startedAt;
    console.info(`[API] ${req.method} ${req.originalUrl || req.url} -> ${res.statusCode} (${durationMs}ms)`);
  });
}

export function createExpressApp() {
  const app = express();
  app.use(cors());
  app.use(express.json());
  app.use((req, res, next) => {
    logApiRequest(req, res);
    next();
  });
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


function callManagerMethod(manager, snakeName, camelName, defaultValue = null) {
  const method = manager?.[snakeName] || manager?.[camelName];
  return typeof method === "function" ? method.call(manager) : defaultValue;
}

function managerAttr(manager, snakeName, camelName, defaultValue = null) {
  return manager?.[snakeName] ?? manager?.[camelName] ?? defaultValue;
}

export function managerAvailableProviders(manager) {
  const providers = callManagerMethod(manager, "get_available_providers", "getAvailableProviders", null);
  if (providers !== null && typeof providers !== "undefined") return Array.from(providers || []);
  return manager?.provider ? [manager.provider] : [];
}

export function managerInitializationMessages(manager) {
  return { ...(managerAttr(manager, "initialization_messages", "initializationMessages", {}) || {}) };
}

export function providerSelectionMap(manager) {
  const available = managerAvailableProviders(manager);
  const sortedProviders = sortProvidersByDisplayOrder(available);
  return Object.fromEntries(sortedProviders.map((provider, index) => [String(index + 1), provider]));
}

export function availableModelIdentifiers(manager, { requireAvailableProvider = true } = {}) {
  const mappings = getIdentifierMappings();
  const availableProviders = new Set(managerAvailableProviders(manager));
  if (!requireAvailableProvider) {
    return ALL_MODEL_IDENTIFIERS.filter((identifier) => mappings[identifier]);
  }
  return ALL_MODEL_IDENTIFIERS.filter((identifier) => mappings[identifier] && availableProviders.has(mappings[identifier].provider));
}

export function modelIdentifiersForProvider(provider) {
  const mappings = getIdentifierMappings();
  return ALL_MODEL_IDENTIFIERS.filter((identifier) => mappings[identifier] && mappings[identifier].provider === provider);
}

export function modelPayload(modelIdentifier, manager, idx = null, { requireAvailableProvider = true, defaultStatus = "Unknown" } = {}) {
  const config = getIdentifierMappings()[modelIdentifier];
  let payloadId = idx;
  if (payloadId === null || typeof payloadId === "undefined") {
    const availableIds = availableModelIdentifiers(manager, { requireAvailableProvider });
    const availableIndex = availableIds.indexOf(modelIdentifier);
    payloadId = availableIndex >= 0 ? availableIndex + 1 : ALL_MODEL_IDENTIFIERS.indexOf(modelIdentifier) + 1;
  }
  const status = managerInitializationMessages(manager)[config.provider] || defaultStatus;
  return {
    id: String(payloadId),
    name: `${config.provider}:${config.model}`,
    display_name: `${config.provider.charAt(0).toUpperCase() + config.provider.slice(1)} (${modelIdentifier})`,
    provider: config.provider,
    selected_model: config.model,
    model: config.model,
    model_identifier: modelIdentifier,
    selected_model_identifier: modelIdentifier,
    model_tier: config.tier,
    selected_model_tier: config.tier,
    strengths: Array.from(config.strengths || []),
    status,
    framework: String(manager?.framework ?? "unknown"),
  };
}

export function modelPayloads(manager, { requireAvailableProvider = true, defaultStatus = "Unknown" } = {}) {
  return availableModelIdentifiers(manager, { requireAvailableProvider })
    .map((identifier, index) => modelPayload(identifier, manager, index + 1, { requireAvailableProvider, defaultStatus }));
}

export function providerPayload(provider, manager, { requireAvailableProvider = true, defaultStatus = "Unknown" } = {}) {
  const modelIdentifiers = modelIdentifiersForProvider(provider);
  return {
    name: provider,
    display_name: getDisplayName(provider),
    provider,
    models: modelIdentifiers.map((identifier) => modelPayload(identifier, manager, null, { requireAvailableProvider, defaultStatus })),
    model_identifiers: modelIdentifiers,
    status: managerInitializationMessages(manager)[provider] || defaultStatus,
  };
}

export function normalizeProviderInput(manager, provider) {
  if (provider === null || typeof provider === "undefined") return null;
  const providerMap = providerSelectionMap(manager);
  const available = new Set(managerAvailableProviders(manager).map((p) => String(p).toLowerCase()));
  const configured = new Set(getAllProviders().map((p) => String(p).toLowerCase()));
  if (typeof provider === "number") return providerMap[String(provider)] ?? null;
  const candidate = String(provider).trim();
  if (!candidate) return null;
  if (candidate in providerMap) return providerMap[candidate];
  const lowered = candidate.toLowerCase();
  if (available.has(lowered) || configured.has(lowered)) return lowered;
  return candidate;
}

export function normalizeModelIdentifierInput(manager, modelIdentifier, { requireAvailableProvider = true } = {}) {
  if (modelIdentifier === null || typeof modelIdentifier === "undefined") return null;
  const mappings = getIdentifierMappings();
  const payloads = modelPayloads(manager, { requireAvailableProvider });
  const modelMap = Object.fromEntries(payloads.map((payload) => [payload.id, payload.model_identifier]));
  const canonicalMap = Object.fromEntries(payloads.map((payload) => [payload.name.toLowerCase(), payload.model_identifier]));
  if (typeof modelIdentifier === "number") return modelMap[String(modelIdentifier)] ?? null;
  const candidate = String(modelIdentifier).trim();
  if (!candidate) return null;
  if (candidate in modelMap) return modelMap[candidate];
  if (candidate in mappings) return candidate;
  const lowered = candidate.toLowerCase();
  if (lowered in canonicalMap) return canonicalMap[lowered];
  return Object.keys(mappings).find((identifier) => {
    const config = mappings[identifier];
    return identifier.toLowerCase() === lowered || config.model.toLowerCase() === lowered;
  }) || null;
}
