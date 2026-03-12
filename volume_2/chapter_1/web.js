import express from "express";
import cors from "cors";

import { normalizeResponseText, toSseLine } from "./stream.js";

export function createWebApi(manager, enableStreaming = false) {
  const app = express();
  app.use(cors());
  app.use(express.json());

  app.get("/", (_, res) => {
    res.json({
      framework: manager.framework || "agent",
      model: manager.model || null,
      streaming_enabled: enableStreaming,
      status: "healthy"
    });
  });

  app.post("/query", async (req, res) => {
    const { topic } = req.body ?? {};
    if (!topic) {
      return res.status(400).json({ error: "Topic is required" });
    }

    const result = await manager.askQuestion(topic);
    if (!result.success) {
      return res.status(500).json({ error: result.error || "Query failed" });
    }
    return res.json(result);
  });

  app.post("/query/stream", async (req, res) => {
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
      for await (const chunk of manager.iterAnswerChunks(topic)) {
        const text = normalizeResponseText(chunk);
        if (text) {
          res.write(toSseLine({ type: "token", token: text }));
        }
      }
      res.write(toSseLine({ type: "done" }));
    } catch (error) {
      res.write(toSseLine({ type: "error", error: error.message }));
    }

    res.end();
  });

  return app;
}

export function runWebServer(manager, host = "0.0.0.0", port = 8000, enableStreaming = false) {
  const app = createWebApi(manager, enableStreaming);
  console.log(`\nStarting API on http://${host}:${port} (streaming=${enableStreaming ? "on" : "off"})`);
  app.listen(port, host);
}
