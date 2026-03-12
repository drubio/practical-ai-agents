import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";

import { chunkText } from "./stream.js";

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
      if (key && !(key in process.env)) {
        process.env[key] = value;
      }
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



const PROVIDER_ENV_KEYS = {
  openai: ["OPENAI_API_KEY"],
  anthropic: ["ANTHROPIC_API_KEY"],
  google: ["GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"],
  google_genai: ["GOOGLE_GENAI_API_KEY", "GOOGLE_API_KEY"],
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

export async function selectStartupModel(modelNames, mode, explicitModel) {
  if (explicitModel) return explicitModel;

  const { buildModels } = await import("./models.js").catch(() => ({ buildModels: null }));
  if (!buildModels) return "openai:gpt-5.2";

  const available = buildModels();
  const names = Array.isArray(modelNames) && modelNames.length ? modelNames : Object.keys(available);
  const catalog = names
    .map((name) => {
      const config = available[name];
      if (!config) return null;
      return {
        name,
        provider: config.provider,
        model: config.model,
        label: modelLabel(config.name, config.model, config.provider)
      };
    })
    .filter(Boolean);

  if (!catalog.length) return "openai:gpt-5.2";

  let configured = catalog.filter((entry) => providerIsConfigured(entry.provider));
  if (!configured.length) configured = catalog;

  let defaultIndex = configured.findIndex((entry) => entry.provider === "openai");
  if (defaultIndex < 0) defaultIndex = 0;

  if (mode !== "cli" || !process.stdin.isTTY || !process.stdout.isTTY) {
    return configured[defaultIndex].model;
  }

  console.log("\nModel selection (configured via environment variables):");
  configured.forEach((entry, idx) => {
    const suffix = idx === defaultIndex ? " [default]" : "";
    console.log(`${idx + 1}. ${entry.label}${suffix}`);
  });

  const selected = await chooseFromList(configured, defaultIndex, `Select model (1-${configured.length}, default ${defaultIndex + 1}): `);
  return selected.model;
}

export function buildCommonArgs(argv = process.argv.slice(2)) {
  let mode = "cli";
  let stream = false;
  let host = "0.0.0.0";
  let port = Number(process.env.PORT || 8000);
  let model = null;

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "cli" || arg === "web") mode = arg;
    else if (arg === "--stream") stream = true;
    else if (arg === "--host") host = argv[i + 1], i += 1;
    else if (arg === "--port") port = Number(argv[i + 1]), i += 1;
    else if (arg === "--model") model = argv[i + 1], i += 1;
  }

  return { mode, stream, host, port, model };
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

  console.log(`\n${manager.toolTriggerHelp || "Tools are selected automatically from your prompt."}`);
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
    console.log("*** Agent working with LLM, awaiting response ***");
    const result = await manager.askQuestion(userInput);
    console.log("\n============= LLM RESPONSE =============");
    console.log(result.success ? result.response : result.error);
    console.log("========================================\n");
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
  yield* chunkText(result.response || "");
}
