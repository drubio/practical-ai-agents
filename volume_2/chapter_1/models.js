import { OpenAI } from "@llamaindex/openai";

export const CHAPTER_1_MODEL_NAMES = ["openai_gpt_5_2"];

export const ALL_MODEL_NAMES = [
  "openai_gpt_4o_mini",
  "openai_gpt_5_2",
  "openai_gpt_4_1",
  "anthropic_claude_3_5_haiku",
  "anthropic_claude_3_5_sonnet",
  "anthropic_claude_3_7_sonnet",
  "google_gemini_1_5_flash",
  "google_gemini_1_5_pro",
  "google_gemini_2_5_flash",
  "xai_grok_3_mini",
  "xai_grok_3",
  "xai_grok_4"
];

export const LLAMAINDEX_MODEL_NAMES = [...ALL_MODEL_NAMES];

export function buildModels() {
  return {
    openai_gpt_4o_mini: { name: "openai_gpt_4o_mini", provider: "openai", model: "openai:gpt-4o-mini" },
    openai_gpt_5_2: { name: "openai_gpt_5_2", provider: "openai", model: "openai:gpt-5.2" },
    openai_gpt_4_1: { name: "openai_gpt_4_1", provider: "openai", model: "openai:gpt-4.1" },
    anthropic_claude_3_5_haiku: { name: "anthropic_claude_3_5_haiku", provider: "anthropic", model: "anthropic:claude-3-5-haiku-latest" },
    anthropic_claude_3_5_sonnet: { name: "anthropic_claude_3_5_sonnet", provider: "anthropic", model: "anthropic:claude-3-5-sonnet-latest" },
    anthropic_claude_3_7_sonnet: { name: "anthropic_claude_sonnet_4_5", provider: "anthropic", model: "anthropic:claude-sonnet-4-5" },
    google_gemini_1_5_flash: { name: "google_gemini_1_5_flash", provider: "google", model: "google:gemini-1.5-flash" },
    google_gemini_1_5_pro: { name: "google_gemini_1_5_pro", provider: "google", model: "google:gemini-1.5-pro" },
    google_gemini_2_5_flash: { name: "google_gemini_2_5_flash", provider: "google_genai", model: "google_genai:gemini-2.5-flash" },
    xai_grok_3_mini: { name: "xai_grok_3_mini", provider: "xai", model: "grok-3-mini" },
    xai_grok_3: { name: "xai_grok_3", provider: "xai", model: "grok-3" },
    xai_grok_4: { name: "xai_grok_4", provider: "xai", model: "grok-4" }
  };
}


export function parseLlamaindexProviderModel(selectedModel) {
  const [provider, model] = selectedModel.includes(":")
    ? selectedModel.split(/:(.+)/)
    : ["openai", selectedModel];
  return { provider, model };
}

export const LLAMAINDEX_PROVIDER_CONFIG = {
  openai: () => ({}),
  xai: () => ({
    baseURL: process.env.XAI_API_BASE || "https://api.x.ai/v1",
    apiKey: process.env.XAI_API_KEY
  }),
  anthropic: () => ({
    baseURL: process.env.ANTHROPIC_OPENAI_BASE_URL,
    apiKey: process.env.ANTHROPIC_API_KEY
  }),
  google: () => ({
    baseURL: process.env.GOOGLE_OPENAI_BASE_URL,
    apiKey: process.env.GOOGLE_API_KEY
  }),
  google_genai: () => ({
    baseURL: process.env.GOOGLE_GENAI_OPENAI_BASE_URL || process.env.GOOGLE_OPENAI_BASE_URL,
    apiKey: process.env.GOOGLE_API_KEY
  })
};

function requireOpenAICompatibleBase(provider, cfg) {
  if (provider === "openai" || provider === "xai") return;
  if (!cfg.baseURL) {
    throw new Error(
      `Provider '${provider}' in JS LlamaIndex requires an OpenAI-compatible base URL env var. ` +
      "Set ANTHROPIC_OPENAI_BASE_URL / GOOGLE_OPENAI_BASE_URL / GOOGLE_GENAI_OPENAI_BASE_URL, or use the Python LlamaIndex scripts for native provider adapters."
    );
  }
}

export function resolveLlamaindexModel(selectedModel) {
  const { provider, model } = parseLlamaindexProviderModel(selectedModel);
  const configBuilder = LLAMAINDEX_PROVIDER_CONFIG[provider];
  if (!configBuilder) {
    throw new Error(`Unsupported provider '${provider}' for '${selectedModel}'. Supported: ${Object.keys(LLAMAINDEX_PROVIDER_CONFIG).join(", ")}`);
  }

  const providerConfig = configBuilder();
  requireOpenAICompatibleBase(provider, providerConfig);

  return {
    provider,
    model,
    llmConfig: {
      model,
      ...(providerConfig.baseURL ? { baseURL: providerConfig.baseURL } : {}),
      ...(providerConfig.apiKey ? { apiKey: providerConfig.apiKey } : {})
    }
  };
}

export function createLlamaindexLLM(selectedModel) {
  const resolved = resolveLlamaindexModel(selectedModel);
  return { ...resolved, llm: new OpenAI(resolved.llmConfig) };
}
