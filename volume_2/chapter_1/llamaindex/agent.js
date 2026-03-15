#!/usr/bin/env node

import * as tools from "../tools.js";
import { ALL_MODEL_IDENTIFIERS, createLlamaindexLLM } from "../models.js";

import {
  buildCommonArgs,
  defaultChunkIterator,
  getChapterLogger,
  collectAsyncEventStreamText,
  logToolCall,
  selectStartupModel,
  runMode
} from "../utils.js";

const logger = getChapterLogger("volume_2.chapter_1.llamaindex.agent");

const TOOLS = {
  summarize_text: logToolCall(logger, "summarize_text", tools.summarizeText)
};

function parseToolCall(text) {
  const match = text.match(/TOOL:\s*([a-z_]+)\s*\nINPUT:\s*([\s\S]*)$/i);
  if (!match) return null;

  const name = match[1].trim();
  const rawInput = match[2].trim();
  if (!(name in TOOLS)) return null;

  try {
    return { name, input: JSON.parse(rawInput) };
  } catch {
    return { name, input: rawInput };
  }
}

class LlamaIndexWorkflowHandler {
  constructor(runPromise) {
    this.runPromise = runPromise;
  }

  async *stream_events() {
    const text = await this.runPromise;
    if (text) {
      yield { delta: text };
    }
  }

  [Symbol.asyncIterator]() {
    return this.stream_events();
  }

  then(onFulfilled, onRejected) {
    return this.runPromise.then(onFulfilled, onRejected);
  }
}

export class LlamaIndexAgentManager {
  framework = "LlamaIndex Agent";
  toolNames = ["summarize_text"];
  toolTriggerHelp =
    "Tools are selected automatically from your prompt; you do not need to type a tool name. " +
    "If you want a specific behavior, ask explicitly (for example: 'summarize this').";

  constructor(model, stream = false) {
    const { provider, model: resolvedModel, llm } = createLlamaindexLLM(model);
    this.provider = provider;
    this.model = resolvedModel;
    this.stream = stream;

    logger.info(
      `Initializing LlamaIndex agent | provider=${this.provider} | model=${this.model} | stream=${this.stream}`
    );

    this.llm = llm;
    this.agent = {
      run: (topic) => new LlamaIndexWorkflowHandler(this.runWithToolLoop(topic))
    };
  }

  async runWithToolLoop(topic) {
    const messages = [
      {
        role: "system",
        content: [
          "You are an AI assistant that can use tools.",
          "Think step-by-step, use tools when needed, and return a concise final answer.",
          "If a tool is needed, respond in this exact format:",
          "TOOL: <tool_name>",
          "INPUT: <valid JSON or plain text>"
        ].join("\n")
      },
      { role: "user", content: topic }
    ];

    for (let attempt = 0; attempt < 4; attempt += 1) {
      const result = await this.llm.chat({ messages });
      const text = String(result?.message?.content || result?.content || "").trim();
      const toolCall = parseToolCall(text);

      if (!toolCall) {
        return text;
      }

      const observation = TOOLS[toolCall.name](toolCall.input);
      messages.push({ role: "assistant", content: text });
      messages.push({ role: "user", content: `Tool result for ${toolCall.name}: ${JSON.stringify(observation)}` });
    }

    throw new Error("Agent exceeded tool-call iteration limit.");
  }

  async askQuestion(topic) {
    try {
      logger.info(`Processing prompt | chars=${topic.length}`);

      const result = this.agent.run(topic);

      const response = this.stream
        ? await collectAsyncEventStreamText(result.stream_events())
        : await collectAsyncEventStreamText(result);

      return {
        success: true,
        stream: this.stream,
        provider: this.provider,
        model: this.model,
        prompt: topic,
        response
      };
    } catch (error) {
      logger.error("LlamaIndex askQuestion failed", error);
      return {
        success: false,
        stream: this.stream,
        provider: this.provider,
        model: this.model,
        prompt: topic,
        error: error?.message || String(error),
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
  const manager = new LlamaIndexAgentManager(startupModel, args.stream);
  await runMode(manager, args.mode, args.host, args.port, args.stream);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
