#!/usr/bin/env node

import * as tools from "../tools.js";
import { ALL_MODEL_IDENTIFIERS, createLlamaindexLLM } from "../utils.js";

import {
  buildTaskPrompt,
  buildCommonArgs,
  defaultChunkIterator,
  getChapterLogger,
  collectAsyncEventStreamText,
  logToolCall,
  selectStartupModel,
  runMode
} from "../utils.js";

const logger = getChapterLogger("volume_2.chapter_1.llamaindex.agent");
const FINAL_RESPONSE_INSTRUCTION = [
  "Treat each non-empty line in the user prompt as a required task and handle every requested step before answering.",
  "You must call calculator for arithmetic expressions or fee calculations, resolve_datetime for scheduling/date phrases, and generate_uuid for unique IDs.",
  'Return your final answer as JSON with this shape: {"tool_calls": [{"name": "<tool_name>", "arguments": {}, "output": "<serialized tool output>"}], "final_answer": "<human readable summary>"}.',
  "Include one tool_calls entry for every tool invocation, in execution order, and store that tool's output or result directly on the same object. Use an empty array when no tools are needed.",
  "Capture the actual tool name, arguments, and output/result exactly as used, and summarize the result for the user in final_answer."
].join(" ");

const TOOLS = {
  calculator: logToolCall(logger, "calculator", tools.calculator),
  resolve_datetime: logToolCall(logger, "resolve_datetime", tools.resolveDatetime),
  generate_uuid: () => logToolCall(logger, "generate_uuid", tools.generateUUID)()
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
  toolNames = ["calculator", "resolve_datetime", "generate_uuid"];
  toolTriggerHelp =
    "Tools are selected automatically from your prompt; you do not need to type a tool name. " +
    "If you want a specific behavior, ask explicitly " +
    "(for example: 'calculate 20 * 5', 'parse tomorrow at 2pm', or 'generate a unique ticket ID').";

  constructor(model, stream = false) {
    const { provider, model: resolvedModel, llm } = createLlamaindexLLM(model);
    this.activeModelIdentifier = model;
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
          "Use the calculator for arithmetic, resolve_datetime for date/time phrases, and generate_uuid when the user asks for a unique ID, UUID, ticket ID, or identifier.",
          FINAL_RESPONSE_INSTRUCTION,
          "Think step-by-step, use tools when needed.",
          "If a tool is needed, respond in this exact format:",
          "TOOL: <tool_name>",
          "INPUT: <valid JSON or plain text>"
        ].join("\n")
      },
      { role: "user", content: buildTaskPrompt(topic) }
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
      messages.push({
        role: "user",
        content: `Tool result for ${toolCall.name}: ${JSON.stringify(observation)}`
      });
    }

    throw new Error("Agent exceeded tool-call iteration limit.");
  }

  async askQuestion(topic) {
    try {
      logger.info(`Received prompt | chars=${topic.length} | multiline=${topic.includes("\n")}`);
      logger.info("Delegating full prompt to LlamaIndex agent");

      const result = this.agent.run(topic);

      logger.info(`Awaiting ${this.stream ? "streamed " : ""}LlamaIndex agent response`);
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
