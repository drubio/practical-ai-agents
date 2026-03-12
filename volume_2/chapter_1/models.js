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
  "xai_grok_2_mini",
  "xai_grok_2",
  "xai_grok_3_beta"
];

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
    xai_grok_2_mini: { name: "xai_grok_2_mini", provider: "xai", model: "xai:grok-2-mini" },
    xai_grok_2: { name: "xai_grok_2", provider: "xai", model: "xai:grok-2" },
    xai_grok_3_beta: { name: "xai_grok_3_beta", provider: "xai", model: "xai:grok-3-beta" }
  };
}
