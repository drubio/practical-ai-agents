#!/usr/bin/env node

import * as z from "zod";
import { createAgent, tool } from "langchain";

import { summarizeText } from "../tools.js";

import {
  buildCommonArgs,
  defaultChunkIterator,
  getChapterLogger,
  logToolCall,
  runMode
} from "../utils.js";

const logger = getChapterLogger("volume_2.chapter_1.langchain.agent");

function buildTools() {
  return [
    tool(logToolCall(logger, "summarize_text", ({ text }) => summarizeText(text)), {
      name: "summarize_text",
      description: "Summarize text.",
      schema: z.object({ text: z.string() })
    })
  ];
}

function extractOutput(result) {
  if (typeof result?.output === "string") return result.output;
  const messages = result?.messages ?? [];
  const content = messages[messages.length - 1]?.content;
  if (typeof content === "string") return content;
  return String(result ?? "");
}

export class LangChainAgentManager {
  framework = "LangChain Agent";
  toolNames = ["summarize_text"];
  toolTriggerHelp = "Tools are selected automatically from your prompt; you do not need to type a tool name.";

  constructor(model = "gpt-5.2") {
    this.model = model;
    logger.info(`Initializing LangChain agent | model=${model}`);
    this.agent = createAgent({
      model,
      tools: buildTools(),
      systemPrompt:
        "You are an AI assistant that can use tools. Think step-by-step, use tools when needed, and return a concise final answer."
    });
  }

  async askQuestion(topic) {
    try {
      logger.info(`Processing prompt | chars=${topic.length}`);
      const result = await this.agent.invoke({ messages: [{ role: "user", content: topic }] });
      return { success: true, provider: "openai", model: this.model, prompt: topic, response: extractOutput(result) };
    } catch (error) {
      logger.error("LangChain askQuestion failed", error);
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
  const manager = new LangChainAgentManager(args.model);
  await runMode(manager, args.mode, args.host, args.port, args.stream);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
