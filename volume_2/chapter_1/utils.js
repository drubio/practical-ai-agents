import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";



export function normalizeResponseText(payload) {
  if (payload == null) return "";

  if (typeof payload === "string") {
    return payload;
  }

  if (typeof payload === "object") {
    for (const key of ["response", "answer", "content", "text", "message"]) {
      const value = payload[key];
      if (typeof value === "string") return value;
    }
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
    if (delayMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
    yield part;
  }
}

export function toSseLine(data) {
  return `data: ${JSON.stringify(data)}\n\n`;
}

export function loadChapterEnv() {
  const chapterRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname));
  const candidates = [
    path.resolve(process.cwd(), ".env"),
    path.resolve(chapterRoot, ".env"),
    path.resolve(chapterRoot, "..", ".env"),
    path.resolve(chapterRoot, "..", "..", ".env")
  ];

  for (const candidate of [...new Set(candidates)]) {
    if (!fs.existsSync(candidate)) continue;

    for (const rawLine of fs.readFileSync(candidate, "utf-8").split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line || line.startsWith("#") || !line.includes("=")) continue;

      const [rawKey, ...rest] = line.split("=");
      const key = rawKey.trim();
      const value = rest.join("=").trim().replace(/^['"]|['"]$/g, "");
      if (key && !(key in process.env)) process.env[key] = value;
    }

    return candidate;
  }

  return null;
}

loadChapterEnv();

export function getChapterLogger(name) {
  const levels = { debug: 10, info: 20, warn: 30, error: 40 };
  const min = levels[(process.env.LOG_LEVEL || "info").toLowerCase()] ?? levels.info;

  const log = (level, message, ...args) => {
    if ((levels[level] ?? 100) < min) return;
    const ts = new Date().toISOString();
    console.log(`${ts} | ${level.toUpperCase()} | ${name} | ${message}`, ...args);
  };

  return {
    debug: (m, ...a) => log("debug", m, ...a),
    info: (m, ...a) => log("info", m, ...a),
    warn: (m, ...a) => log("warn", m, ...a),
    error: (m, ...a) => log("error", m, ...a)
  };
}

export function logToolCall(logger, toolName, fn) {
  return (arg) => {
    logger.info(`Tool call | name=${toolName} | input=%o`, arg);
    const result = fn(arg);
    logger.info(`Tool result | name=${toolName} | output=%o`, result);
    return result;
  };
}

function extractTextFromContent(content) {
  if (content == null) return "";
  if (typeof content === "string") return content;

  if (Array.isArray(content)) {
    const parts = [];
    for (const item of content) {
      if (typeof item === "string") {
        parts.push(item);
        continue;
      }

      if (item && typeof item === "object") {
        if (item.type === "text" && typeof item.text === "string") {
          parts.push(item.text);
          continue;
        }
        if (typeof item.text === "string") {
          parts.push(item.text);
          continue;
        }
        if (typeof item.delta === "string") {
          parts.push(item.delta);
        }
      }
    }
    return parts.join("");
  }

  if (typeof content === "object") {
    if (typeof content.text === "string") return content.text;
    if (typeof content.delta === "string") return content.delta;
  }

  return "";
}

function extractTextFromMessageLike(obj) {
  if (obj == null) return "";
  if (typeof obj === "string") return obj;

  const direct = extractTextFromContent(obj);
  if (direct) return direct;

  for (const key of ["content", "text", "response", "message", "output", "delta"]) {
    const value = obj?.[key];
    const text = extractTextFromContent(value);
    if (text) return text;

    if (value != null && value !== obj) {
      const nested = extractTextFromMessageLike(value);
      if (nested) return nested;
    }
  }

  if (typeof obj === "object") {
    for (const key of ["output", "response", "content", "text", "delta"]) {
      const text = extractTextFromContent(obj[key]);
      if (text) return text;
    }

    const messages = obj.messages || [];
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const text = extractTextFromMessageLike(messages[i]);
      if (text) return text;
    }
  }

  return "";
}

function appendStreamText(parts, text) {
  if (!text) return;

  const current = parts.join("");
  if (!current) {
    parts.push(text);
    return;
  }

  if (text.startsWith(current)) {
    const suffix = text.slice(current.length);
    if (suffix) parts.push(suffix);
    return;
  }

  if (text === current || text === parts[parts.length - 1]) return;
  parts.push(text);
}

function collectLangchainStreamText(result) {
  const parts = [];

  for (const chunk of result || []) {
    if (!Array.isArray(chunk) || chunk.length !== 2) continue;
    const [streamMode, payload] = chunk;
    if (streamMode !== "messages") continue;

    const messageChunk = Array.isArray(payload) ? payload[0] : payload;
    if (!messageChunk) continue;

    const text = extractTextFromMessageLike(messageChunk);
    if (text) parts.push(text);
  }

  return parts.join("").trim();
}

export async function collectAsyncEventStreamText(eventStream) {
  const parts = [];
  for await (const event of eventStream) {
    if (Array.isArray(event) && event.length === 2 && typeof event[0] === "string") {
      const [streamMode, payload] = event;
      if (streamMode !== "messages") {
        continue;
      }

      const messageChunk = Array.isArray(payload) ? payload[0] : payload;
      const messageText = extractTextFromMessageLike(messageChunk);
      if (messageText) {
        appendStreamText(parts, messageText);
      }
      continue;
    }

    const delta = event?.delta;
    if (typeof delta === "string" && delta) {
      appendStreamText(parts, delta);
      continue;
    }

    const text = extractTextFromMessageLike(event);
    if (text) appendStreamText(parts, text);
  }
  return parts.join("").trim();
}

export async function extractStreamText(result) {
  if (result == null) return "";
  if (typeof result === "string") return result.trim();

  if (typeof result?.[Symbol.asyncIterator] === "function") {
    return collectAsyncEventStreamText(result);
  }

  if (typeof result?.stream_events === "function") {
    return collectAsyncEventStreamText(result.stream_events());
  }

  return collectLangchainStreamText(result);
}

export async function extractOutputText(result) {
  if (result == null) return "";
  if (typeof result === "string") return result.trim();

  const text = extractTextFromMessageLike(result);
  if (text) return text.trim();

  if (typeof result?.then === "function") {
    const resolved = await result;
    return extractOutputText(resolved);
  }

  return String(result);
}

export async function renderResponseText(rawResponse, stream) {
  if (stream) return extractStreamText(rawResponse);
  return extractOutputText(rawResponse);
}

const PROVIDER_ENV_KEYS = {
  openai: ["OPENAI_API_KEY"],
  anthropic: ["ANTHROPIC_API_KEY"],
  google: ["GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"],
  google_genai: ["GOOGLE_GENAI_API_KEY", "GOOGLE_API_KEY"],
  "google-genai": ["GOOGLE_GENAI_API_KEY", "GOOGLE_API_KEY"],
  xai: ["XAI_API_KEY"]
};

function providerIsConfigured(provider) {
  const keys = PROVIDER_ENV_KEYS[provider] ?? [];
  return keys.some((key) => (process.env[key] || "").trim());
}

function modelLabel(name, model, provider) {
  const providerLabel = provider.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
  return `${name} (${providerLabel}, ${model})`;
}

async function chooseFromList(options, defaultIndex, prompt) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const ask = (q) => new Promise((resolve) => rl.question(q, resolve));

  try {
    while (true) {
      const raw = (await ask(prompt)).trim();
      if (!raw) return options[defaultIndex];
      const choice = Number(raw) - 1;
      if (Number.isInteger(choice) && choice >= 0 && choice < options.length) return options[choice];
      console.log("Invalid selection. Please try again.");
    }
  } finally {
    rl.close();
  }
}

export async function selectStartupModel(modelIdentifiers, mode, explicitModelIdentifier) {
  if (explicitModelIdentifier) return explicitModelIdentifier;

  const { getIdentifierMappings } = await import("./models.js").catch(() => ({ getIdentifierMappings: null }));
  if (!getIdentifierMappings) throw new Error("Model registry unavailable for startup selection.");

  const available = getIdentifierMappings();
  const names = Array.isArray(modelIdentifiers) && modelIdentifiers.length ? modelIdentifiers : Object.keys(available);
  const catalog = names
    .map((name) => {
      const config = available[name];
      if (!config) return null;
      return {
        name,
        provider: config.provider,
        model: config.model,
        identifier: name,
        label: modelLabel(config.name, config.model, config.provider)
      };
    })
    .filter(Boolean);

  if (!catalog.length) throw new Error("No model configurations available for startup selection.");

  let configured = catalog.filter((entry) => providerIsConfigured(entry.provider));
  if (!configured.length) configured = catalog;

  const defaultIndex = 0;

  if (mode !== "cli" || !process.stdin.isTTY || !process.stdout.isTTY) {
    return configured[defaultIndex].identifier;
  }

  console.log("\nModel selection (configured via environment variables):");
  configured.forEach((entry, idx) => {
    const suffix = idx === defaultIndex ? " [default]" : "";
    console.log(`${idx + 1}. ${entry.label}${suffix}`);
  });

  const selected = await chooseFromList(
    configured,
    defaultIndex,
    `Select model (1-${configured.length}, default ${defaultIndex + 1}): `
  );
  return selected.identifier;
}

export function buildCommonArgs(argv = process.argv.slice(2)) {
  let mode = "cli";
  let stream = false;
  let host = "0.0.0.0";
  let port = Number(process.env.PORT || 8000);
  let modelIdentifier = null;

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "cli" || arg === "web") mode = arg;
    else if (arg === "--stream") stream = true;
    else if (arg === "--host") host = argv[i + 1], i += 1;
    else if (arg === "--port") port = Number(argv[i + 1]), i += 1;
    else if (arg === "--model-identifier") modelIdentifier = argv[i + 1], i += 1;
  }

  return { mode, stream, host, port, modelIdentifier };
}

function printCliBanner(manager) {
  console.log(`\n===== ${manager.framework} CLI =====`);
  console.log("Type a question and press Enter.");
  console.log("Type 'exit' to quit.\n");

  const names = Array.isArray(manager.toolNames) ? manager.toolNames : [];
  if (names.length) {
    console.log("Available local tools:");
    names.forEach((name) => console.log(`  - ${name}`));
  } else {
    console.log("Available local tools: (none declared)");
  }

  console.log(`\n${manager.toolTriggerHelp || "Tools are selected automatically from your prompt. You can mention a specific task (for example: calculate) to encourage tool use."}`);
  console.log("Tip: multi-line pasted text is accepted as a single prompt.");
  console.log("====================================");
}

export async function runInteractiveCli(manager) {
  printCliBanner(manager);
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const ask = (prompt) => new Promise((resolve) => rl.question(prompt, resolve));

  while (true) {
    const userInput = (await ask("\n(exit or enter question) > ")).trim();

    if (!userInput) {
      console.log("No prompt provided. Please enter a question.");
      continue;
    }
    if (["exit", "quit"].includes(userInput.toLowerCase())) {
      console.log("Exiting.");
      break;
    }

    console.log("*** Agent is working locally ***");
    const result = await manager.askQuestion(userInput);
    if (result.localOnly) {
      console.log("*** Local tool response generated (LLM bypassed) ***");
    } else {
      console.log("*** Agent working with LLM, awaiting response ***");
    }
    console.log("\n============= AGENT RESPONSE =============");
    if (result.success) {
      console.log(await renderResponseText(result.response, manager.stream));
    } else {
      console.log(result.error || "");
    }
    console.log("==========================================\n");
  }

  rl.close();
}

export async function runMode(manager, mode, host, port, stream) {
  if (mode === "web") {
    const { runWebServer } = await import("./web.js");
    runWebServer(manager, host, port, stream);
    return;
  }
  await runInteractiveCli(manager);
}

export async function* defaultChunkIterator(manager, topic) {
  const result = await manager.askQuestion(topic);

  let responseText = "";
  if (result.success) {
    responseText = await renderResponseText(result.response, manager.stream);
  } else {
    responseText = String(result.error || "");
  }

  yield* chunkText(responseText);
}
