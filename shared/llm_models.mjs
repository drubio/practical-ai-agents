import { existsSync, readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

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

export const UNSUPPORTED_MODEL_PARAMETERS = {
  openai_gpt_5_4_pro: ["temperature", "top_p"],
  openai_gpt_5_mini: ["temperature"],
};

export const TEMPERATURE_UNSUPPORTED_MODEL_IDENTIFIERS = new Set(
  Object.entries(UNSUPPORTED_MODEL_PARAMETERS)
    .filter(([, parameters]) => parameters.includes("temperature"))
    .map(([identifier]) => identifier),
);

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
    deepseek_4_pro: { name: "deepseek_4_pro", provider: "deepseek", model: "deepseek-v4-pro", tier: "advanced", strengths: ["deep-analysis", "reasoning", "long-form"] },
    deepseek_4_flash: { name: "deepseek_4_flash", provider: "deepseek", model: "deepseek-v4-flash", tier: "lite", strengths: ["social", "fast", "cost-efficient"] },    
  };
}

export function getModelConfig(selectedModel) {
  const config = getIdentifierMappings()[selectedModel];
  if (!config) throw new Error(`Unknown model identifier '${selectedModel}'`);
  return config;
}

export function getUnsupportedModelParameters(selection) {
  const config = typeof selection === "string" ? resolveModelConfig(selection) : selection;
  return UNSUPPORTED_MODEL_PARAMETERS[config.name] ?? [];
}

export function modelSupportsTemperature(selection) {
  return !getUnsupportedModelParameters(selection).includes("temperature");
}

export function getModelsForProvider(provider) {
  const mappings = getIdentifierMappings();
  return ALL_MODEL_IDENTIFIERS
    .filter((identifier) => mappings[identifier]?.provider === provider)
    .map((identifier) => mappings[identifier]);
}

export function getProviderModelConfig(provider) {
  const [config] = getModelsForProvider(provider);
  if (!config) throw new Error(`No model identifiers configured for provider '${provider}'`);
  return config;
}

export function getProviderModelIdentifier(provider) {
  return getProviderModelConfig(provider).name;
}

export function resolveModelIdentifier(selection, availableProviders = null) {
  const available = availableProviders ? new Set(availableProviders) : null;
  const mappings = getIdentifierMappings();
  if (selection && mappings[selection]) {
    const { provider } = mappings[selection];
    return !available || available.has(provider) ? selection : null;
  }
  if (selection && PROVIDER_DISPLAY_NAMES[selection]) {
    return !available || available.has(selection) ? getProviderModelIdentifier(selection) : null;
  }
  if (selection) return null;
  if (available?.size) {
    const [provider] = sortProvidersByDisplayOrder([...available]);
    return getProviderModelIdentifier(provider);
  }
  return null;
}

export function resolveModelConfig(selection) {
  const mappings = getIdentifierMappings();
  if (mappings[selection]) return mappings[selection];
  if (PROVIDER_DISPLAY_NAMES[selection]) return getProviderModelConfig(selection);
  throw new Error(`Unknown model or provider selection '${selection}'`);
}


export function getApiKey(provider) {
  for (const envVar of PROVIDER_API_KEY_ENV_VARS[provider] || []) {
    const value = (process.env[envVar] || "").trim();
    if (value) return value;
  }
  return null;
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
