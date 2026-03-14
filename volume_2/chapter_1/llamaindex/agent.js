#!/usr/bin/env node

import * as tools from "../tools.js";
import { ALL_MODEL_IDENTIFIERS, createLlamaindexLLM } from "../models.js";

import {
  buildCommonArgs,
  defaultChunkIterator,
  extractOutputText,
  getChapterLogger,
  logToolCall,
  selectStartupModel,
  runMode
} from "../utils.js";

const logger = getChapterLogger("volume_2.chapter_1.llamaindex.agent");

const TOOL_MAP = {
  summarize_text: logToolCall(logger, "summarize_text", tools.summarizeText)
};


function parseToolCall(text) {
  const match = text.match(/TOOL:\s*([a-z_]+)\s*\nINPUT:\s*([\s\S]*)$/i);
  if (!match) return null;

  const name = match[1].trim();
  const rawInput = match[2].trim();
  if (!(name in TOOL_MAP)) return null;

  try {
    return { name, input: JSON.parse(rawInput) };
  } catch {
    return { name, input: rawInput };
  }
}

export class LlamaIndexAgentManager {
  framework = "LlamaIndex Agent";
  toolNames = Object.keys(TOOL_MAP);
  toolTriggerHelp = "Tools are selected automatically from your prompt; you do not need to type a tool name.";

  constructor(model) {
    const { provider, model: resolvedModel, llm } = createLlamaindexLLM(model);
    this.provider = provider;
    this.model = resolvedModel;
    logger.info(`Initializing LlamaIndex agent | provider=${this.provider} | model=${this.model}`);
    this.llm = llm;
  }

  async askQuestion(topic) {
    try {
      logger.info(`Processing prompt | chars=${topic.length}`);

      const messages = [
        {
          role: "system",
          content: [
            "You are an AI assistant that can use tools.",
            "When needed, reply strictly in this format and nothing else:",
            "TOOL: <tool_name>",
            "INPUT: <valid JSON or plain text>",
            "If no tool is needed, return a concise final answer directly."
          ].join("\n")
        },
        { role: "user", content: topic }
      ];

      for (let attempt = 0; attempt < 4; attempt += 1) {
        const result = await this.llm.chat({ messages });
        const text = extractOutputText(result).trim();
        const toolCall = parseToolCall(text);

        if (!toolCall) {
          return { success: true, provider: this.provider, model: this.model, prompt: topic, response: text };
        }

        const observation = TOOL_MAP[toolCall.name](toolCall.input);
        messages.push({ role: "assistant", content: text });
        messages.push({ role: "user", content: `Tool result for ${toolCall.name}: ${JSON.stringify(observation)}` });
      }

      return {
        success: false,
        provider: this.provider,
        model: this.model,
        prompt: topic,
        error: "Agent exceeded tool-call iteration limit.",
        response: null
      };
    } catch (error) {
      logger.error("LlamaIndex askQuestion failed", error);
      return {
        success: false,
        provider: this.provider,
        model: this.model,
        prompt: topic,
        error: error.message,
        response: null
      };
    }
  }

  async *iterAnswerChunks(topic) {
    yield* defaultChunkIterator(this, topic);
  }
}

async function main() {
  const args = buildCommonArgs();
  const startupModel = await selectStartupModel(ALL_MODEL_IDENTIFIERS, args.mode, args.modelIdentifier);
  const manager = new LlamaIndexAgentManager(startupModel);
  await runMode(manager, args.mode, args.host, args.port, args.stream);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
