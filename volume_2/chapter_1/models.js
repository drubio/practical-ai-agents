import { Anthropic } from "@llamaindex/anthropic";
import { Gemini } from "@llamaindex/google";
import { OpenAI } from "@llamaindex/openai";

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
];


export function getIdentifierMappings() {
  return {
    openai_gpt_5_4_pro: { name: "openai_gpt_5_4", provider: "openai", model: "gpt-5.4-pro" },
    openai_gpt_5_4: { name: "openai_gpt_5_4", provider: "openai", model: "gpt-5.4" }, 
    openai_gpt_5_mini: { name: "openai_gpt_4o_mini", provider: "openai", model: "gpt-5-mini" },
    anthropic_claude_opus_4_6: { name: "anthropic_claude_opus_4_6", provider: "anthropic", model: "claude-opus-4-6" },
    anthropic_claude_sonnet_4_6: { name: "anthropic_claude_sonnet_4_6", provider: "anthropic", model: "claude-sonnet-4-6" },
    anthropic_claude_haiku_4_5: { name: "anthropic_claude_haiku_4_5", provider: "anthropic", model: "claude-haiku-4-5" },
    google_genai_gemini_3_1_pro: { name: "google_genai_gemini_3_1_pro", provider: "google-genai", model: "gemini-3.1-pro-preview" },
    google_genai_gemini_3_flash: { name: "google_genai_gemini_3_flash", provider: "google-genai", model: "gemini-3-flash-preview" },
    google_genai_gemini_3_1_flash_lite: { name: "google_genai_gemini_3_1_flash_lite", provider: "google-genai", model: "gemini-3.1-flash-lite-preview" },
    xai_grok_4: { name: "xai_grok_4", provider: "xai", model: "grok-4" },
    xai_grok_3: { name: "xai_grok_3", provider: "xai", model: "grok-3" },      
    xai_grok_3_mini: { name: "xai_grok_3_mini", provider: "xai", model: "grok-3-mini" }

  };
}

function inferProvider(promptLower, selectedTools) {
  const tools = new Set(selectedTools.map((name) => String(name).toLowerCase()));

  const socialMarkers = ["social", "tweet", "x.com", "reddit", "viral", "engagement", "thread", "post"];
  const codingMarkers = ["code", "bug", "debug", "refactor", "typescript", "python", "sql", "api"];
  const researchMarkers = ["research", "paper", "citation", "study", "benchmark", "literature", "compare"];

  if (["chatgpt", "openai", "gpt"].some((token) => promptLower.includes(token))) return "openai";
  if (["claude", "anthropic"].some((token) => promptLower.includes(token))) return "anthropic";
  if (["gemini", "google"].some((token) => promptLower.includes(token))) return "google-genai";
  if (["grok", "xai"].some((token) => promptLower.includes(token))) return "xai";

  if (socialMarkers.some((token) => promptLower.includes(token))) return "xai";

  const calculatorIsPrimary = tools.has("calculator") && tools.size <= 2;
  if (codingMarkers.some((token) => promptLower.includes(token)) || calculatorIsPrimary) return "anthropic";

  if (researchMarkers.some((token) => promptLower.includes(token))) return "google-genai";

  return "openai";
}

function inferTier(promptLower, selectedTools) {
  const tools = new Set(selectedTools.map((name) => String(name).toLowerCase()));

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
  const provider = inferProvider(promptLower, selectedTools || []);
  const tier = inferTier(promptLower, selectedTools || []);

  const providerTierCandidates = {
    openai: {
      advanced: "openai_gpt_5_4_pro",
      standard: "openai_gpt_5_4",
      lite: "openai_gpt_5_mini"
    },
    anthropic: {
      advanced: "anthropic_claude_opus_4_6",
      standard: "anthropic_claude_sonnet_4_6",
      lite: "anthropic_claude_haiku_4_5"
    },
    "google-genai": {
      advanced: "google_genai_gemini_3_1_pro",
      standard: "google_genai_gemini_3_flash",
      lite: "google_genai_gemini_3_1_flash_lite"
    },
    xai: {
      advanced: "xai_grok_4",
      standard: "xai_grok_3",
      lite: "xai_grok_3_mini"
    }
  };

  const fallbackOrder = [tier, "standard", "advanced", "lite"];
  const mappings = getIdentifierMappings();
  const candidates = providerTierCandidates[provider] || providerTierCandidates.openai;

  for (const tierName of fallbackOrder) {
    const candidateName = candidates[tierName];
    if (selectedPool.has(candidateName)) {
      return mappings[candidateName];
    }
  }

  if (selectedPool.size) {
    const [firstAvailable] = selectedPool;
    return mappings[firstAvailable];
  }

  const [firstMappedKey] = Object.keys(mappings);
  return mappings[firstMappedKey];
}


export function parseLlamaindexProviderModel(selectedModel) {
  const [provider, model] = selectedModel.includes(":")
    ? selectedModel.split(/:(.+)/)
    : [null, selectedModel];
  return { provider, model };
}

export const LLAMAINDEX_PROVIDER_CONFIG = {
  openai: () => ({
    llmClass: OpenAI,
    llmConfig: {
      model: null
    }
  }),
  xai: () => ({
    llmClass: OpenAI,
    llmConfig: {
      model: null,
      baseURL: process.env.XAI_API_BASE || "https://api.x.ai/v1",
      apiKey: process.env.XAI_API_KEY
    }
  }),
  anthropic: () => ({
    llmClass: Anthropic,
    llmConfig: {
      model: null,
      apiKey: process.env.ANTHROPIC_API_KEY
    }
  }),
  google: () => ({
    llmClass: Gemini,
    llmConfig: {
      model: null,
      apiKey: process.env.GOOGLE_API_KEY || process.env.GOOGLE_GENAI_API_KEY
    }
  }),
  google_genai: () => ({
    llmClass: Gemini,
    llmConfig: {
      model: null,
      apiKey: process.env.GOOGLE_GENAI_API_KEY || process.env.GOOGLE_API_KEY
    }
  }),
  "google-genai": () => ({
    llmClass: Gemini,
    llmConfig: {
      model: null,
      apiKey: process.env.GOOGLE_GENAI_API_KEY || process.env.GOOGLE_API_KEY
    }
  })
};

export function resolveLlamaindexModel(selectedModel) {
  const available = getIdentifierMappings();
  const fromIdentifier = available[selectedModel];
  const parsed = parseLlamaindexProviderModel(selectedModel);
  const provider = fromIdentifier?.provider ?? parsed.provider;
  const model = fromIdentifier?.model ?? parsed.model;

  if (!provider) {
    throw new Error(`Unsupported model '${selectedModel}'. Use a model identifier from ALL_MODEL_IDENTIFIERS or provider:model syntax.`);
  }

  const configBuilder = LLAMAINDEX_PROVIDER_CONFIG[provider];
  if (!configBuilder) {
    throw new Error(`Unsupported provider '${provider}' for '${selectedModel}'. Supported: ${Object.keys(LLAMAINDEX_PROVIDER_CONFIG).join(", ")}`);
  }

  const providerConfig = configBuilder();
  const llmClass = providerConfig.llmClass;
  const llmConfig = {
    model,
    ...Object.fromEntries(
      Object.entries(providerConfig.llmConfig || {})
        .filter(([key, value]) => key !== "model" && value != null && value !== "")
    )
  };

  return {
    provider,
    model,
    llmClass,
    llmConfig
  };
}

export function createLlamaindexLLM(selectedModel) {
  const resolved = resolveLlamaindexModel(selectedModel);
  return { ...resolved, llm: new resolved.llmClass(resolved.llmConfig) };
}
