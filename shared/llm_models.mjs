import { existsSync, readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

import { Anthropic } from "@llamaindex/anthropic";
import { Gemini } from "@llamaindex/google";
import { OpenAI } from "@llamaindex/openai";

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

export const ALL_MODEL_IDENTIFIERS = [
  "openai_gpt_5_4_pro",
  "openai_gpt_5_4",
  "openai_gpt_5_mini",
  "anthropic_claude_opus_4_6",
  "anthropic_claude_sonnet_4_6",
  "anthropic_claude_haiku_4_5",
  "google_genai_gemini_3_1_pro",
  "google_genai_gemini_3_flash",
  "google_genai_gemini_3_1_flash_lite",
  "xai_grok_4",
  "xai_grok_3",
  "xai_grok_3_mini",
  "deepseek_4_flash",
  "deepseek_4_pro",  
];

export const PROVIDER_DISPLAY_NAMES = {
  openai: "OpenAI GPT",
  anthropic: "Anthropic Claude",
  google: "Google Gemini",
  xai: "xAI Grok",
  deepseek: "DeepSeek",
};

export const PROVIDER_API_KEY_ENV_VARS = {
  openai: ["OPENAI_API_KEY"],
  anthropic: ["ANTHROPIC_API_KEY"],
  google: ["GOOGLE_GENAI_API_KEY", "GOOGLE_API_KEY"],
  xai: ["XAI_API_KEY"],
  deepseek: ["DEEPSEEK_API_KEY"],
};

export const PROVIDER_DEFAULT_MODEL_IDENTIFIERS = {
  openai: "openai_gpt_5_4",
  anthropic: "anthropic_claude_sonnet_4_6",
  google: "google_genai_gemini_3_flash",
  xai: "xai_grok_4",
  deepseek: "deepseek_4_flash",  
};

export function getIdentifierMappings() {
  return {
    openai_gpt_5_4_pro: { name: "openai_gpt_5_4_pro", provider: "openai", model: "gpt-5.4-pro", tier: "advanced", strengths: ["precision", "analysis", "long-form"] },
    openai_gpt_5_4: { name: "openai_gpt_5_4", provider: "openai", model: "gpt-5.4", tier: "standard", strengths: ["balanced", "reasoning", "general"] },
    openai_gpt_5_mini: { name: "openai_gpt_5_mini", provider: "openai", model: "gpt-5-mini", tier: "lite", strengths: ["cost-efficient", "fast", "general"] },
    anthropic_claude_opus_4_6: { name: "anthropic_claude_opus_4_6", provider: "anthropic", model: "claude-opus-4-6", tier: "advanced", strengths: ["deep-reasoning", "coding", "planning"] },
    anthropic_claude_sonnet_4_6: { name: "anthropic_claude_sonnet_4_6", provider: "anthropic", model: "claude-sonnet-4-6", tier: "standard", strengths: ["coding", "reasoning", "balanced"] },
    anthropic_claude_haiku_4_5: { name: "anthropic_claude_haiku_4_5", provider: "anthropic", model: "claude-haiku-4-5", tier: "lite", strengths: ["speed", "coding", "cost-efficient"] },
    google_genai_gemini_3_1_pro: { name: "google_genai_gemini_3_1_pro", provider: "google", model: "gemini-3.1-pro-preview", tier: "advanced", strengths: ["reasoning", "tool-use", "multimodal"] },
    google_genai_gemini_3_flash: { name: "google_genai_gemini_3_flash", provider: "google", model: "gemini-3-flash-preview", tier: "standard", strengths: ["long-context", "research", "synthesis"] },
    google_genai_gemini_3_1_flash_lite: { name: "google_genai_gemini_3_1_flash_lite", provider: "google", model: "gemini-3.1-flash-lite-preview", tier: "lite", strengths: ["fast", "retrieval", "classification"] },
    xai_grok_4: { name: "xai_grok_4", provider: "xai", model: "grok-4", tier: "advanced", strengths: ["deep-analysis", "social", "long-form"] },
    xai_grok_3: { name: "xai_grok_3", provider: "xai", model: "grok-3", tier: "standard", strengths: ["social", "trends", "analysis"] },
    xai_grok_3_mini: { name: "xai_grok_3_mini", provider: "xai", model: "grok-3-mini", tier: "lite", strengths: ["social", "fast", "cost-efficient"] },
    deepseek_4_pro: { name: "deepseek_4_pro", provider: "deepseek", model: "deepseek-v4-pro", tier: "advanced", strengths: ["deep-analysis", "social", "long-form"] },
    deepseek_4_pro: { name: "deepseek_4_pro", provider: "deepseek", model: "deepseek-v4-pro", tier: "standard", strengths: ["social", "trends", "analysis"] },
    deepseek_4_flash: { name: "deepseek_4_flash", provider: "deepseek", model: "deepseek-v4-flash", tier: "lite", strengths: ["social", "fast", "cost-efficient"] },    
  };
}

export function getDefaultModelConfig(provider) {
  const identifier = PROVIDER_DEFAULT_MODEL_IDENTIFIERS[provider];
  return getIdentifierMappings()[identifier];
}

export function getProviderCatalog(providers = Object.keys(PROVIDER_DISPLAY_NAMES)) {
  return Object.fromEntries(
    providers.map((provider) => {
      const config = getDefaultModelConfig(provider);
      return [provider, {
        provider,
        canonicalProvider: config.provider,
        apiKeyEnv: PROVIDER_API_KEY_ENV_VARS[provider][0],
        defaultModel: config.model,
        defaultModelIdentifier: config.name,
        displayName: PROVIDER_DISPLAY_NAMES[provider] || provider,
      }];
    })
  );
}

export function getApiKey(provider) {
  for (const envVar of PROVIDER_API_KEY_ENV_VARS[provider] || []) {
    const value = (process.env[envVar] || "").trim();
    if (value) return value;
  }
  return null;
}

export function getDefaultModelName(provider) {
  return getDefaultModelConfig(provider).model;
}

export function getDisplayName(provider) {
  return PROVIDER_DISPLAY_NAMES[provider] || provider;
}

export function getPublicProviderNames() {
  return Object.keys(PROVIDER_DISPLAY_NAMES);
}

export function getAllProviders() {
  return getPublicProviderNames();
}

export function sortProvidersByDisplayOrder(providers) {
  const displayRank = new Map(Object.keys(PROVIDER_DISPLAY_NAMES).map((provider, index) => [provider, index]));
  return [...providers].sort((a, b) => {
    const aRank = displayRank.has(a) ? displayRank.get(a) : displayRank.size;
    const bRank = displayRank.has(b) ? displayRank.get(b) : displayRank.size;
    if (aRank !== bRank) return aRank - bRank;
    return getDisplayName(a).localeCompare(getDisplayName(b)) || String(a).localeCompare(String(b));
  });
}

function inferProvider(promptLower, selectedTools) {
  const tools = new Set((selectedTools || []).map((name) => String(name).toLowerCase()));
  const socialMarkers = ["social", "tweet", "x.com", "reddit", "viral", "engagement", "thread", "post"];
  const codingMarkers = ["code", "bug", "debug", "refactor", "typescript", "python", "sql", "api"];
  const researchMarkers = ["research", "paper", "citation", "study", "benchmark", "literature", "compare"];
  if (["chatgpt", "openai", "gpt"].some((token) => promptLower.includes(token))) return "openai";
  if (["claude", "anthropic"].some((token) => promptLower.includes(token))) return "anthropic";
  if (["gemini", "google"].some((token) => promptLower.includes(token))) return "google";
  if (["grok", "xai"].some((token) => promptLower.includes(token))) return "xai";
  if (["deepseek"].some((token) => promptLower.includes(token))) return "deepseek";
  if (socialMarkers.some((token) => promptLower.includes(token))) return "xai";
  const calculatorIsPrimary = tools.has("calculator") && tools.size <= 2;
  if (codingMarkers.some((token) => promptLower.includes(token)) || calculatorIsPrimary) return "anthropic";
  if (researchMarkers.some((token) => promptLower.includes(token)) || tools.has("parse_content")) return "google";
  return "openai";
}

function inferTier(promptLower, selectedTools) {
  const tools = new Set((selectedTools || []).map((name) => String(name).toLowerCase()));
  const advancedTools = new Set(["analyze_text", "extract_tasks", "route_workflow", "summarize_text"]);
  const liteTools = new Set(["calculator", "resolve_datetime", "extract_keywords", "score_priority"]);
  const advancedMarkers = ["architecture", "multi-step", "deep", "strategy", "tradeoff", "production design", "root cause", "long-form", "thorough"];
  const liteMarkers = ["quick", "brief", "short", "one-liner", "cheap", "fast"];
  if ([...tools].some((name) => advancedTools.has(name))) return "advanced";
  if (tools.size > 0 && [...tools].every((name) => liteTools.has(name))) return "lite";
  if (advancedMarkers.some((token) => promptLower.includes(token)) || promptLower.length > 900) return "advanced";
  if (liteMarkers.some((token) => promptLower.includes(token))) return "lite";
  return "standard";
}

export function routeModelForPrompt(prompt, selectedTools, modelIdentifiers = ALL_MODEL_IDENTIFIERS) {
  const promptLower = String(prompt || "").toLowerCase();
  const selectedPool = new Set(modelIdentifiers || ALL_MODEL_IDENTIFIERS);
  const provider = inferProvider(promptLower, selectedTools);
  const tier = inferTier(promptLower, selectedTools);
  const providerTierCandidates = {
    openai: { advanced: "openai_gpt_5_4_pro", standard: "openai_gpt_5_4", lite: "openai_gpt_5_mini" },
    anthropic: { advanced: "anthropic_claude_opus_4_6", standard: "anthropic_claude_sonnet_4_6", lite: "anthropic_claude_haiku_4_5" },
    google: { advanced: "google_genai_gemini_3_1_pro", standard: "google_genai_gemini_3_flash", lite: "google_genai_gemini_3_1_flash_lite" },
    xai: { advanced: "xai_grok_4", standard: "xai_grok_3", lite: "xai_grok_3_mini" },
    deepseek: { advanced: "deepseek_4_pro", standard: "deepseek_4_pro", lite: "deepseek_4_flash" },    
  };
  const mappings = getIdentifierMappings();
  const candidates = providerTierCandidates[provider] || providerTierCandidates.openai;
  for (const tierName of [tier, "standard", "lite", "advanced"]) {
    const candidateName = candidates[tierName];
    if (selectedPool.has(candidateName)) return mappings[candidateName];
  }
  if (selectedPool.size) {
    const [firstAvailable] = selectedPool;
    return mappings[firstAvailable];
  }
  const [firstMappedKey] = Object.keys(mappings);
  return mappings[firstMappedKey];
}


export function resolveLlamaindexModel(selectedModel) {
  const config = getIdentifierMappings()[selectedModel];
  if (!config) {
    throw new Error(`Unknown model identifier '${selectedModel}'`);
  }

  const provider = config.provider;
  const llmProvider = provider;
  const builders = {
    openai: () => ({ llmClass: OpenAI, llmConfig: { model: config.model } }),
    anthropic: () => ({ llmClass: Anthropic, llmConfig: { model: config.model, apiKey: process.env.ANTHROPIC_API_KEY } }),
    google: () => ({ llmClass: Gemini, llmConfig: { model: config.model, apiKey: process.env.GOOGLE_GENAI_API_KEY || process.env.GOOGLE_API_KEY } }),
    xai: () => ({ llmClass: OpenAI, llmConfig: { model: config.model, apiKey: process.env.XAI_API_KEY, baseURL: process.env.XAI_API_BASE || "https://api.x.ai/v1" } }),
    deepseek: () => ({ llmClass: OpenAI, llmConfig: { model: config.model, apiKey: process.env.DEEPSEEK_API_KEY, baseURL: process.env.DEEPSEEK_API_BASE || "https://api.deepseek.com" } }),    
  };
  const builder = builders[llmProvider];
  if (!builder) {
    throw new Error(`Unsupported provider '${config.provider}' for '${selectedModel}'`);
  }
  return { provider: config.provider, model: config.model, ...builder() };
}

export function createLlamaindexLLM(selectedModel) {
  const resolved = resolveLlamaindexModel(selectedModel);
  return { ...resolved, llm: new resolved.llmClass(resolved.llmConfig) };
}
