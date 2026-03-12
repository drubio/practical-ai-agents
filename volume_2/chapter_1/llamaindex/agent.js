#!/usr/bin/env node

import { OpenAI } from "@llamaindex/openai";

import * as tools from "../tools.js";

import {
  buildCommonArgs,
  defaultChunkIterator,
  getChapterLogger,
  logToolCall,
  runMode
} from "../utils.js";

const logger = getChapterLogger("volume_2.chapter_1.llamaindex.agent");

const TOOL_MAP = {
  summarize_text: logToolCall(logger, "summarize_text", tools.summarizeText)
};

function extractText(result) {
  const content = result?.message?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((block) => block?.type === "text" && typeof block?.text === "string")
      .map((block) => block.text)
      .join("\n");
  }
  return String(content ?? result ?? "");
}

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

  constructor(model = "gpt-5.2") {
    this.model = model;
    logger.info(`Initializing LlamaIndex agent | model=${model}`);
    this.llm = new OpenAI({ model });
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
        const text = extractText(result).trim();
        const toolCall = parseToolCall(text);

        if (!toolCall) {
          return { success: true, provider: "openai", model: this.model, prompt: topic, response: text };
        }

        const observation = TOOL_MAP[toolCall.name](toolCall.input);
        messages.push({ role: "assistant", content: text });
        messages.push({ role: "user", content: `Tool result for ${toolCall.name}: ${JSON.stringify(observation)}` });
      }

      return {
        success: false,
        provider: "openai",
        model: this.model,
        prompt: topic,
        error: "Agent exceeded tool-call iteration limit.",
        response: null
      };
    } catch (error) {
      logger.error("LlamaIndex askQuestion failed", error);
      return {
        success: false,
        provider: "openai",
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
  const manager = new LlamaIndexAgentManager(args.model);
  await runMode(manager, args.mode, args.host, args.port, args.stream);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
