import express from "express";
import cors from "cors";

import { chunkText, getProviderOptions, iterStreamTextChunks, normalizeAgentApiPayload, toSseLine } from "./utils.js";

function supportsHistory(manager) {
  return typeof manager?.getHistory === "function";
}

function supportsResetMemory(manager) {
  return typeof manager?.resetMemory === "function";
}

async function buildQueryResponse(result, request, manager, streamOverride = undefined) {
  const streamMode = streamOverride ?? Boolean(manager.stream);
  const normalized = await normalizeAgentApiPayload(result.response, streamMode);
  const effectiveSessionId = request.sessionId || request.session_id || "default";

  const responsePayload = {
    success: true,
    framework: manager.framework || "agent",
    provider: result.provider,
    model: result.model,
    response: normalized.response,
    raw_answer: normalized.raw_answer,
    prompt: result.prompt || request.template?.replace("{topic}", request.topic) || request.topic,
    parameters: {
      temperature: request.temperature,
      max_tokens: request.max_tokens,
      template: request.template || "{topic}"
    },
    session_id: result.session_id || effectiveSessionId
  };

  if (Array.isArray(normalized.tool_calls)) {
    responsePayload.tool_calls = normalized.tool_calls;
  }

  return responsePayload;
}

async function managerProviderOptions(manager) {
  if (typeof manager?.getProviderOptions === "function") {
    return await manager.getProviderOptions();
  }
  return getProviderOptions(manager.modelIdentifiers);
}

async function resolveRequestManager(manager, requestedModel) {
  const autoSelectionValues = new Set([undefined, null, "", manager.autoProviderOptionName, manager.activeModelIdentifier, manager.model]);
  if (autoSelectionValues.has(requestedModel)) {
    return { manager, activeModelIdentifier: manager.activeModelIdentifier || requestedModel };
  }

  const availableIdentifiers = new Set(Array.isArray(manager.modelIdentifiers) ? manager.modelIdentifiers : []);
  if (availableIdentifiers.size && !availableIdentifiers.has(requestedModel)) {
    const error = new Error(`Unsupported provider/model selection '${requestedModel}'`);
    error.statusCode = 400;
    throw error;
  }

  return {
    manager: new manager.constructor(requestedModel, Boolean(manager.stream)),
    activeModelIdentifier: requestedModel
  };
}

export function createWebApi(manager, enableStreaming = false) {
  const app = express();
  app.use(cors());
  app.use(express.json());

  app.get("/", (_, res) => {
    res.json({
      framework: manager.framework || "agent",
      available_providers: [manager.provider || "unknown"],
      total_available: 1,
      initialization_status: {
        [manager.provider || "unknown"]: "Ready"
      },
      status: "healthy"
    });
  });

  app.get("/capabilities", (_, res) => {
    res.json({
      framework: manager.framework || "agent",
      streaming: true,
      memory: supportsHistory(manager) && supportsResetMemory(manager),
      memory_retrieval: false,
      coagent: false
    });
  });

  app.get("/providers", async (_, res) => {
    const providers = await managerProviderOptions(manager);
    res.json({
      framework: manager.framework || "agent",
      providers,
      count: providers.length
    });
  });

  app.post("/query", async (req, res) => {
    const { topic } = req.body ?? {};
    if (!topic) {
      return res.status(400).json({ error: "Topic is required" });
    }

    try {
      const { manager: requestManager, activeModelIdentifier } = await resolveRequestManager(manager, req.body?.provider);
      const result = await requestManager.askQuestion(topic);
      if (!result?.success) {
        return res.status(400).json({ detail: result?.error || "Query failed" });
      }
      result.active_model_identifier = activeModelIdentifier;
      return res.json(await buildQueryResponse(result, req.body ?? {}, requestManager));
    } catch (error) {
      return res.status(error.statusCode || 500).json({ detail: error.message || "Query failed" });
    }
  });

  const streamHandler = async (req, res) => {
    if (!enableStreaming) {
      return res.status(404).json({ detail: "Streaming is disabled. Start with --stream." });
    }

    const { topic } = req.body ?? {};
    if (!topic) {
      return res.status(400).json({ error: "Topic is required" });
    }

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    try {
      const { manager: requestManager, activeModelIdentifier } = await resolveRequestManager(manager, req.body?.provider);
      const result = await requestManager.askQuestion(topic);
      if (!result?.success) {
        res.write(toSseLine({ type: "error", error: result?.error || "Query failed" }));
        return res.end();
      }

      let streamedText = "";
      if (requestManager.stream) {
        for await (const chunk of iterStreamTextChunks(result.response)) {
          if (chunk) {
            streamedText += chunk;
            res.write(toSseLine({ type: "chunk", content: chunk }));
          }
        }
      } else {
        const normalized = await normalizeAgentApiPayload(result.response, false);
        const fullText = normalized.raw_answer || "";
        for (const chunk of chunkText(fullText)) {
          if (chunk) {
            streamedText += chunk;
            res.write(toSseLine({ type: "chunk", content: chunk }));
          }
        }
      }

      const normalized = await normalizeAgentApiPayload(streamedText || result.response, false);
      const donePayload = {
        type: "done",
        provider: result.provider,
        model: result.model,
        active_model_identifier: activeModelIdentifier,
        response: normalized.response && typeof normalized.response === "object" ? normalized.response : null,
        raw_answer: normalized.raw_answer,
        session_id: result.session_id || req.body?.sessionId || req.body?.session_id || "default"
      };
      if (Array.isArray(normalized.tool_calls)) {
        donePayload.tool_calls = normalized.tool_calls;
      }
      res.write(toSseLine(donePayload));
    } catch (error) {
      res.write(toSseLine({ type: "error", error: error.message || "Streaming failed" }));
    }

    res.end();
  };

  app.post("/query-stream", streamHandler);
  app.post("/query/stream", streamHandler);

  app.post("/query-all", (_, res) => {
    res.status(400).json({ detail: "This agent server supports only single-provider query mode" });
  });

  app.get("/history", (req, res) => {
    if (!supportsHistory(manager)) {
      return res.status(400).json({ detail: "Session memory not supported by this manager" });
    }
    return res.json(manager.getHistory(req.query.provider || "openai", req.query.session_id || "default"));
  });

  app.post("/reset-memory", (req, res) => {
    if (!supportsResetMemory(manager)) {
      return res.status(400).json({ detail: "Session memory not supported by this manager" });
    }
    const provider = req.body?.provider ?? req.query?.provider;
    const sessionId = req.body?.sessionId ?? req.body?.session_id ?? req.query?.session_id;
    return res.json(manager.resetMemory(provider, sessionId));
  });

  return app;
}

export function runWebServer(manager, host = "0.0.0.0", port = 8000, enableStreaming = false) {
  const app = createWebApi(manager, enableStreaming);
  console.log(`Starting web server for ${manager.framework || "Unknown"}`);
  console.log(`Docs: http://${host}:${port}/`);
  console.log(`Health: http://${host}:${port}/`);
  app.listen(port, host);
}
