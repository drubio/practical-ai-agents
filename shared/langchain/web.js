import express from 'express';
import cors from 'cors';
import { normalizeResponseText } from '../utils.mjs';
import { modelPayloads, normalizeModelIdentifierInput, toSseLine } from '../web.mjs';

function extractText(result) {
  return normalizeResponseText(result?.finalText ?? result?.final_text ?? result?.response ?? '');
}

function requestModelIdentifier(manager, body = {}) {
  const requestedModelIdentifier = body.modelIdentifier ?? body.model_identifier ?? null;
  return normalizeModelIdentifierInput(manager, requestedModelIdentifier ?? body.provider, { requireAvailableProvider: false });
}

function managerForModel(manager, modelIdentifier) {
  if (!modelIdentifier || modelIdentifier === manager?.modelIdentifier) return manager;

  if (typeof manager?.constructor === 'function') {
    try {
      return new manager.constructor(modelIdentifier);
    } catch {}
  }

  if (typeof manager?.setModelIdentifier === 'function') manager.setModelIdentifier(modelIdentifier);
  else if (typeof manager?.setModel === 'function') manager.setModel(modelIdentifier);
  return manager;
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
    const providers = modelPayloads(manager, { requireAvailableProvider: false, defaultStatus: '✓ Initialized successfully' });
    const activeProvider = providers[0]?.name ?? null;
    res.json({ providers, count: providers.length, active_provider: activeProvider });
  });

  app.get('/capabilities', (_req, res) => {
    res.json({ streaming: true, default_stream: streamDefault, stream_enabled: Boolean(manager?.stream), single_provider: true });
  });

  app.post('/query', async (req, res) => {
    try {
      const topic = req.body?.topic;
      const selectedModelIdentifier = requestModelIdentifier(manager, req.body);
      const selectedProvider = selectedModelIdentifier ?? req.body?.provider ?? null;
      if (!topic) return res.status(400).json({ error: 'Topic is required' });
      const requestManager = managerForModel(manager, selectedModelIdentifier);
      const result = await requestManager.askQuestion(topic);
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
      const selectedModelIdentifier = requestModelIdentifier(manager, req.body);
      const requestManager = managerForModel(manager, selectedModelIdentifier);
      const result = await requestManager.askQuestion(topic);

      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      if (!result?.success) {
        res.write(toSseLine({ type: 'error', error: result?.error || 'Query failed' }));
        return res.end();
      }

      const tokens = extractText(result).split(/\s+/).filter(Boolean);
      for (const token of tokens) {
        res.write(toSseLine({ type: 'chunk', content: token }));
      }
      res.write(toSseLine({ type: 'done' }));
      return res.end();
    } catch (error) {
      if (!res.headersSent) return res.status(500).json({ error: error?.message || String(error) });
      res.write(toSseLine({ type: 'error', error: error?.message || String(error) }));
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
