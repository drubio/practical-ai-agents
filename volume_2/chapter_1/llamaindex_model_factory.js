import { Anthropic } from "@llamaindex/anthropic";
import { Gemini } from "@llamaindex/google";
import { OpenAI } from "@llamaindex/openai";

import { getIdentifierMappings, normalizeProvider } from "../../shared/llm_models.mjs";

export function resolveLlamaindexModel(selectedModel) {
  const config = getIdentifierMappings()[selectedModel];
  if (!config) {
    throw new Error(`Unknown model identifier '${selectedModel}'`);
  }

  const provider = normalizeProvider(config.provider);
  const llmProvider = provider === "google_genai" ? "google" : provider;
  const builders = {
    openai: () => ({ llmClass: OpenAI, llmConfig: { model: config.model } }),
    anthropic: () => ({ llmClass: Anthropic, llmConfig: { model: config.model, apiKey: process.env.ANTHROPIC_API_KEY } }),
    google: () => ({ llmClass: Gemini, llmConfig: { model: config.model, apiKey: process.env.GOOGLE_GENAI_API_KEY || process.env.GOOGLE_API_KEY } }),
    xai: () => ({ llmClass: OpenAI, llmConfig: { model: config.model, apiKey: process.env.XAI_API_KEY, baseURL: process.env.XAI_API_BASE || "https://api.x.ai/v1" } }),
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
