import express from 'express';
import cors from 'cors';
import { ALL_MODEL_IDENTIFIERS, getIdentifierMappings } from '../../shared/llm_models.mjs';

function extractText(result) {
  return String(result?.finalText ?? result?.response ?? '');
}

function providerPayload(manager, modelIdentifier, idx) {
  const mappings = getIdentifierMappings();
  const cfg = mappings[modelIdentifier] || {};
  const provider = String(cfg.provider ?? manager?.provider ?? 'unknown');
  const model = String(cfg.model ?? manager?.model ?? modelIdentifier);
  const name = `${provider}:${model}`;
  return {
    id: String(idx),
    name,
    display_name: `${provider.charAt(0).toUpperCase() + provider.slice(1)} (${modelIdentifier})`,
    provider,
    default_model: model,
    model,
    model_identifier: modelIdentifier,
    status: '✓ Initialized successfully',
    framework: String(manager?.framework ?? 'unknown'),
  };
}

export function createWebApi(manager, streamDefault = false) {
  const app = express();
  app.use(cors());
  app.use(express.json());

  app.get('/', (_req, res) => {
    res.json({ framework: manager.framework || 'unknown', tool_names: manager.toolNames || [], status: 'healthy' });
  });

  app.get('/health', (_req, res) => {
    res.json({ status: 'healthy', framework: manager.framework || 'unknown' });
  });

  app.get('/providers', (_req, res) => {
    const providers = ALL_MODEL_IDENTIFIERS.map((mid, idx) => providerPayload(manager, mid, idx + 1));
    const activeProvider = providers[0]?.name ?? 'unknown';
    res.json({ providers, count: providers.length, active_provider: activeProvider });
  });

  app.get('/capabilities', (_req, res) => {
    res.json({ streaming: true, default_stream: streamDefault, single_provider: true });
  });

  app.post('/query', async (req, res) => {
    try {
      const topic = req.body?.topic;
      const selectedProvider = req.body?.provider ?? req.body?.model_identifier ?? null;
      if (!topic) return res.status(400).json({ error: 'Topic is required' });
      const result = await manager.askQuestion(topic);
      if (!result?.success) return res.status(400).json({ error: result?.error || 'Query failed' });
      return res.json({ success: true, framework: manager.framework || 'unknown', topic, selected_provider: selectedProvider, response: extractText(result), raw: result });
    } catch (error) {
      return res.status(500).json({ error: error?.message || String(error) });
    }
  });

  app.post('/query-stream', async (req, res) => {
    try {
      const topic = req.body?.topic;
      if (!topic) return res.status(400).json({ error: 'Topic is required' });
      const result = await manager.askQuestion(topic);

      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      if (!result?.success) {
        res.write(`data: ${JSON.stringify({ type: 'error', error: result?.error || 'Query failed' })}\n\n`);
        return res.end();
      }

      const tokens = extractText(result).split(/\s+/).filter(Boolean);
      for (const token of tokens) {
        res.write(`data: ${JSON.stringify({ type: 'chunk', content: token })}\n\n`);
      }
      res.write(`data: ${JSON.stringify({ type: 'done' })}\n\n`);
      return res.end();
    } catch (error) {
      if (!res.headersSent) return res.status(500).json({ error: error?.message || String(error) });
      res.write(`data: ${JSON.stringify({ type: 'error', error: error?.message || String(error) })}\n\n`);
      return res.end();
    }
  });

  return app;
}

export async function runWebServer(manager, host = '0.0.0.0', port = 8000, streamDefault = false) {
  const app = createWebApi(manager, streamDefault);
  app.listen(port, host, () => {
    console.log(`Starting web server for ${manager.framework || 'unknown'}...`);
    console.log(`Health check: http://${host}:${port}/health`);
  });
}
