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
    google_genai_gemini_3_1_pro: { name: "google_genai_gemini_3_1_pro", provider: "google_genai", model: "gemini-3.1-pro-preview" },
    google_genai_gemini_3_flash: { name: "google_genai_gemini_3_flash", provider: "google_genai", model: "gemini-3-flash-preview" },
    google_genai_gemini_3_1_flash_lite: { name: "google_genai_gemini_3_1_flash_lite", provider: "google_genai", model: "gemini-3.1-flash-lite-preview" },
    xai_grok_4: { name: "xai_grok_4", provider: "xai", model: "grok-4" },
    xai_grok_3: { name: "xai_grok_3", provider: "xai", model: "grok-3" },      
    xai_grok_3_mini: { name: "xai_grok_3_mini", provider: "xai", model: "grok-3-mini" }

  };
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
