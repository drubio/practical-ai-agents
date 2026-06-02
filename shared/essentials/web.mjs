import {
  normalizeResponseText,
  parseStructuredJsonResponse,
} from "../utils.mjs";
import { getIdentifierMappings } from "../llm_models.mjs";
import { withSelectedModelIdentifier } from "./utils.mjs";
import {
  availableModelIdentifiers,
  buildManager,
  captureConsoleOutputAsync,
  createExpressApp,
  modelIdentifiersForProvider,
  modelPayload,
  modelPayloads,
  normalizeModelIdentifierInput,
  normalizeProviderInput,
  providerPayload as buildProviderPayload,
  providerSelectionMap,
  resolveSessionId,
  resultIsSuccess,
  streamTextSse,
  supportsCoagent,
  supportsMemory,
  supportsMemoryRetrieval,
  supportsSessionMemory,
  toSseLine,
} from "../web.mjs";

export {
  availableModelIdentifiers,
  buildProviderPayload,
  modelIdentifiersForProvider,
  modelPayload,
  modelPayloads,
  normalizeModelIdentifierInput,
  normalizeProviderInput,
  providerSelectionMap,
};

export function parseStructuredRawResponse(rawResponse) {
  if (typeof rawResponse === "undefined" || rawResponse === null) return null;
  try {
    const parsed = parseStructuredJsonResponse(rawResponse);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export function extractAnswerText(responseObj, rawResponse) {
  if (responseObj && typeof responseObj === "object") {
    const answer = responseObj.answer || responseObj.distilled || responseObj.summary;
    if (typeof answer === "string" && answer.trim()) return answer;
  }
  return normalizeResponseText(rawResponse);
}

export function recoverStructuredParseError(result) {
  if (!result) return result;
  const rawResponse = result.rawResponse;
  const parsedResponse = parseStructuredRawResponse(rawResponse);
  if (result.success) {
    if (parsedResponse && result.response && typeof result.response === "object") {
      const metadataNotes = result.response?.metadata?.notes;
      if (typeof metadataNotes === "string" && metadataNotes.startsWith("Failed to parse structured JSON response")) {
        return { ...result, response: parsedResponse, rawAnswer: extractAnswerText(parsedResponse, rawResponse) };
      }
    }
    return result;
  }
  const errorMessage = String(result.error || "");
  if (!errorMessage.includes("Failed to parse structured JSON response") || !rawResponse) return result;
  const normalized = normalizeResponseText(rawResponse);
  const recoveredResponse = parsedResponse || { answer: normalized, distilled: normalized, metadata: { confidence: "low", notes: errorMessage } };
  return { ...result, success: true, error: null, response: recoveredResponse, rawAnswer: extractAnswerText(recoveredResponse, rawResponse) };
}

export async function askQuestionWithSession(manager, topic, provider, template, maxTokens, temperature, sessionId) {
  if (supportsSessionMemory(manager)) {
    return manager.askQuestion(topic, provider, template, maxTokens, temperature, sessionId);
  }
  return manager.askQuestion(topic, provider, template, maxTokens, temperature);
}

export function buildQueryPayload(manager, body = {}) {
  const {
    topic,
    provider = null,
    model_identifier = null,
    modelIdentifier = null,
    template = "{topic}",
    max_tokens = 1000,
    temperature = 0.7,
    session_id = "default",
    sessionId = null,
  } = body;
  const effectiveSessionId = resolveSessionId(sessionId, session_id);
  let selectedModelIdentifier = normalizeModelIdentifierInput(manager, modelIdentifier ?? model_identifier);
  if (!selectedModelIdentifier) selectedModelIdentifier = normalizeModelIdentifierInput(manager, provider);
  const resolvedProvider = selectedModelIdentifier
    ? getIdentifierMappings()[selectedModelIdentifier].provider
    : normalizeProviderInput(manager, provider);
  return {
    topic,
    provider: resolvedProvider,
    modelIdentifier: selectedModelIdentifier,
    template,
    maxTokens: max_tokens,
    temperature,
    sessionId: effectiveSessionId,
  };
}

export async function executeManagerQuery(manager, queryPayload) {
  const { topic, provider, template, maxTokens, temperature, sessionId, modelIdentifier } = queryPayload;
  return captureConsoleOutputAsync(async () => recoverStructuredParseError(
    await withSelectedModelIdentifier(modelIdentifier, async () => askQuestionWithSession(manager, topic, provider, template, maxTokens, temperature, sessionId)),
  ));
}

export function queryErrorPayload(result, logs = "") {
  return {
    error: result?.error || "Query failed",
    provider: result?.provider,
    debug: logs || null,
  };
}

export function serializeQueryResponse(manager, queryPayload, result, logs = "") {
  return {
    success: true,
    framework: manager.framework,
    topic: queryPayload.topic,
    selected_provider: queryPayload.modelIdentifier || queryPayload.provider,
    provider: result.provider,
    model: result.model,
    model_identifier: queryPayload.modelIdentifier || result.modelIdentifier || result.model_identifier || null,
    response: (typeof result.response === "object" && result.response !== null) ? result.response : normalizeResponseText(result.response),
    parameters: {
      temperature: result.temperature,
      max_tokens: result.maxTokens,
      template: queryPayload.template,
    },
    prompt: result.prompt,
    session_id: result.sessionId || queryPayload.sessionId,
    ...(logs ? { debug: logs } : {}),
  };
}

export function serializeDoneEvent(result, sessionId) {
  return {
    type: "done",
    provider: result.provider,
    model: result.model,
    model_identifier: result.modelIdentifier ?? result.model_identifier ?? null,
    response: (typeof result.response === "object" && result.response !== null) ? result.response : null,
    token_usage: result.tokenUsage ?? result.token_usage ?? null,
    session_id: result.sessionId ?? sessionId,
  };
}

export function createWebApi(managerClassOrFactory) {
  const app = createExpressApp();
  let manager;
  const initPromise = (async () => {
    manager = buildManager(managerClassOrFactory);
    await manager._checkProviders();
    return manager;
  })();
  app.use(async (_, __, next) => {
    if (!manager) await initPromise;
    next();
  });

  app.get("/", (_, res) => {
    const available = manager.getAvailableProviders();
    res.json({
      framework: manager.framework,
      available_providers: available,
      available_provider_details: available.map((provider) => buildProviderPayload(provider, manager)),
      available_model_details: modelPayloads(manager),
      total_available: available.length,
      total_models_available: modelPayloads(manager).length,
      initialization_status: manager.initializationMessages,
      status: available.length > 0 ? "healthy" : "no_providers",
    });
  });
  app.get("/providers", (_, res) => {
    const available = manager.getAvailableProviders();
    const models = modelPayloads(manager);
    res.json({
      framework: manager.framework,
      providers: models,
      provider_groups: available.map((provider) => buildProviderPayload(provider, manager)),
      available_providers: available,
      models,
      count: models.length,
      provider_count: available.length,
      model_count: models.length,
      active_provider: models[0]?.name ?? null,
    });
  });
  app.get("/capabilities", (_, res) => {
    res.json({ framework: manager.framework, streaming: true, memory: supportsMemory(manager), memory_retrieval: supportsMemoryRetrieval(manager), coagent: supportsCoagent(manager) });
  });

  app.post("/query", async (req, res) => {
    try {
      const queryPayload = buildQueryPayload(manager, req.body);
      if (!queryPayload.topic) return res.status(400).json({ error: "Topic is required" });
      const { result, logs } = await executeManagerQuery(manager, queryPayload);
      if (result && queryPayload.modelIdentifier) result.modelIdentifier = queryPayload.modelIdentifier;
      if (!resultIsSuccess(result)) return res.status(400).json(queryErrorPayload(result, logs));
      return res.json(serializeQueryResponse(manager, queryPayload, result, logs));
    } catch (error) {
      return res.status(500).json({ error: error.message, framework: manager.framework });
    }
  });

  app.post("/query-stream", async (req, res) => {
    try {
      const queryPayload = buildQueryPayload(manager, req.body);
      if (!queryPayload.topic) return res.status(400).json({ error: "Topic is required" });
      const { result } = await executeManagerQuery(manager, queryPayload);
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      if (result && queryPayload.modelIdentifier) result.modelIdentifier = queryPayload.modelIdentifier;
      if (!resultIsSuccess(result)) {
        res.write(toSseLine({ type: "error", error: result?.error || "Query failed" }));
        return res.end();
      }
      for await (const event of streamTextSse(normalizeResponseText(result.response), 28, 35)) {
        res.write(event);
      }
      res.write(toSseLine(serializeDoneEvent(result, queryPayload.sessionId)));
      return res.end();
    } catch (error) {
      if (!res.headersSent) return res.status(500).json({ error: error.message, framework: manager.framework });
      res.write(toSseLine({ type: "error", error: error.message }));
      return res.end();
    }
  });

  app.get("/history", async (req, res) => {
    if (!supportsSessionMemory(manager)) return res.status(400).json({ error: "Session memory not supported by this manager" });
    const { provider = "openai", session_id = "default", sessionId = null } = req.query;
    return res.json(await Promise.resolve(manager.getHistory(normalizeProviderInput(manager, provider) || "openai", sessionId ?? session_id ?? "default")));
  });

  app.post("/reset-memory", async (req, res) => {
    if (!supportsSessionMemory(manager)) return res.status(400).json({ error: "Session memory not supported by this manager" });
    const body = req.body || {};
    const provider = body.provider ?? req.query?.provider ?? null;
    const resetSessionId = body.sessionId ?? body.session_id ?? req.query?.sessionId ?? req.query?.session_id ?? null;
    return res.json(await Promise.resolve(manager.resetMemory(normalizeProviderInput(manager, provider), resetSessionId)));
  });

  app.get("/health", (_, res) => {
    const available = manager.getAvailableProviders();
    res.json({ status: available.length > 0 ? "healthy" : "unhealthy", framework: manager.framework, providers_available: available.length });
  });

  return app;
}

export async function runWebServer(managerClassOrFactory, host = "0.0.0.0", port = 8000) {
  const app = createWebApi(managerClassOrFactory);
  let frameworkName = "Unknown";
  try { frameworkName = buildManager(managerClassOrFactory).framework; } catch {}
  app.listen(port, host, () => {
    console.log(`Starting web server for ${frameworkName} framework...`);
    console.log(`Health check: http://${host}:${port}/health`);
    console.log(`Status: http://${host}:${port}/`);
  });
}
