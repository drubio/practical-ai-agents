#!/usr/bin/env node

import * as z from "../../chapter_1/node_modules/zod/index.js";
import { createAgent, tool } from "../../chapter_1/node_modules/langchain/dist/index.js";

import * as tools from "../../chapter_1/tools.js";
import {
  buildTaskPrompt,
  buildCommonArgs,
  defaultChunkIterator,
  getChapterLogger,
  logToolCall,
  describeModelAvailability,
  getRoutableModelIdentifiers,
  runMode
} from "../../chapter_1/utils.js";
import { ALL_MODEL_IDENTIFIERS, routeModelForPrompt } from "../../chapter_1/utils.js";

const logger = getChapterLogger("volume_2.chapter_2.langchain.agent_routing");
const FINAL_RESPONSE_INSTRUCTION =
  "Treat each non-empty line in the user prompt as a required task and handle every requested step before answering. " +
  "You must call calculator for arithmetic expressions or fee calculations, " +
  "resolve_datetime for scheduling/date phrases, and generate_uuid for unique IDs. " +
  'Return your final answer as JSON with this shape: {"tool_calls": [{"name": "<tool_name>", "arguments": {}, "output": "<serialized tool output>"}], "final_answer": "<human readable summary>"}. ' +
  "Include one tool_calls entry for every tool invocation, in execution order, and store that tool's output or result directly on the same object. Use an empty array when no tools are needed. " +
  "Capture the actual tool name, arguments, and output/result exactly as used, and summarize the result for the user in final_answer.";

export const ALL_TOOL_NAMES = ["calculator", "resolve_datetime", "generate_uuid"];
export const AUTO_PROVIDER_OPTION = {
  name: "auto",
  display_name: "Model auto-selected by agent based on prompt",
  provider: "agent",
  model: "auto",
  status: "Ready"
};

export function buildTools(logToolCallFn, activeLogger) {
  return {
    calculator: tool(logToolCallFn(activeLogger, "calculator", ({ expression }) => tools.calculator(expression)), {
      name: "calculator",
      description: "Evaluate expression.",
      schema: z.object({ expression: z.string() })
    }),
    resolve_datetime: tool(logToolCallFn(activeLogger, "resolve_datetime", ({ text }) => tools.resolveDatetime(text)), {
      name: "resolve_datetime",
      description: "Resolve datetime from text.",
      schema: z.object({ text: z.string() })
    }),
    generate_uuid: tool(logToolCallFn(activeLogger, "generate_uuid", () => tools.generateUUID()), {
      name: "generate_uuid",
      description: "Generate unique UUID.",
      schema: z.object({})
    }),
  };
}

export function selectTools(logToolCallFn, activeLogger, toolNames) {
  const available = buildTools(logToolCallFn, activeLogger);
  return toolNames.map((name) => available[name]);
}

export class LangChainAgentRoutingManager {
  framework = "LangChain Agent Routing";
  toolNames = ALL_TOOL_NAMES;
  autoProviderOptionName = AUTO_PROVIDER_OPTION.name;
  modelIdentifiers = [];
  toolTriggerHelp =
    "Tools are selected automatically from your prompt; you do not need to type a tool name. If you want a specific behavior, ask explicitly (for example: 'calculate 20 * 5' or 'parse tomorrow at 2pm').";

  constructor(modelIdentifiers, initialModel, stream = true) {
    this.modelIdentifiers = modelIdentifiers;
    this.modelIdentifier = initialModel;
    this.provider = "unknown";
    this.model = initialModel;
    this.stream = stream;
    this.activeModelIdentifier = this.autoProviderOptionName;
    this.agentCache = new Map();
    logger.info(
      `Initializing LangChain routing agent | provider=${this.provider} | initial_model=${this.model} | stream=${this.stream}`
    );
  }

  async getProviderOptions() {
    return [{ ...AUTO_PROVIDER_OPTION }];
  }

  getAgent(provider, model, selectedToolNames) {
    const key = `${provider}:${model}:${selectedToolNames.join(",")}`;
    if (!this.agentCache.has(key)) {
      logger.info(`Building LangChain agent | provider=${provider} | model=${model} | tools=${selectedToolNames.join(",")}`);
      this.agentCache.set(
        key,
        createAgent({
          model: `${provider}:${model}`,
          tools: selectTools(logToolCall, logger, selectedToolNames),
          systemPrompt:
            "You are an AI assistant that can use tools. " +
            "Choose the best tool(s) among those provided. " +
            FINAL_RESPONSE_INSTRUCTION
        })
      );
    }
    return this.agentCache.get(key);
  }

  async askQuestion(topic) {
    try {
      logger.info(`Received prompt | chars=${topic.length} | multiline=${topic.includes("\n")}`);
      logger.info("Delegating full prompt to routed LangChain agent");
      const selectedToolNames = ALL_TOOL_NAMES;
      const selectedModel = routeModelForPrompt(topic, selectedToolNames, this.modelIdentifiers);
      this.provider = selectedModel.provider;
      this.model = selectedModel.model;
      this.modelIdentifier = selectedModel.name;
      this.activeModelIdentifier = this.autoProviderOptionName;

      const agent = this.getAgent(selectedModel.provider, selectedModel.model, selectedToolNames);
      const input = { messages: [{ role: "user", content: buildTaskPrompt(topic) }] };
      logger.info(`Awaiting ${this.stream ? "streamed " : ""}LangChain agent response`);
      const result = this.stream
        ? await agent.stream(input, { streamMode: ["messages", "updates"] })
        : await agent.invoke(input);

      return {
        success: true,
        stream: this.stream,
        provider: selectedModel.provider,
        model: selectedModel.model,
        modelName: selectedModel.name,
        modelTier: selectedModel.tier,
        selectedTools: selectedToolNames,
        prompt: topic,
        response: result
      };
    } catch (error) {
      logger.error("LangChain askQuestion failed", error);
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
  const modelIdentifiers = await getRoutableModelIdentifiers(ALL_MODEL_IDENTIFIERS, args.modelIdentifier);
  const manager = new LangChainAgentRoutingManager(modelIdentifiers, modelIdentifiers[0], args.stream);
  manager.toolTriggerHelp = `${manager.toolTriggerHelp} ${await describeModelAvailability(ALL_MODEL_IDENTIFIERS)}`;
  await runMode(manager, args.mode, args.host, args.port, args.stream);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
