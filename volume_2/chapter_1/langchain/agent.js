#!/usr/bin/env node

import { createAgent } from "langchain";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

import * as tools from "../tools.js";
import {
  buildCommonArgs,
  defaultChunkIterator,
  getChapterLogger,
  logToolCall,
  selectStartupModel,
  runMode
} from "../utils.js";
import { ALL_MODEL_IDENTIFIERS, getIdentifierMappings } from "../models.js";

const logger = getChapterLogger("volume_2.chapter_1.langchain.agent");

const calculatorTool = tool(
  ({ expression }) => logToolCall(logger, "calculator", tools.calculator)(expression),
  {
    name: "calculator",
    description: "Safely evaluate arithmetic expressions.",
    schema: z.object({ expression: z.string() })
  }
);

const resolveDatetimeTool = tool(
  ({ text }) => logToolCall(logger, "resolve_datetime", tools.resolveDatetime)(text),
  {
    name: "resolve_datetime",
    description: "Resolve date/time phrases.",
    schema: z.object({ text: z.string() })
  }
);

const generateUuidTool = tool(
  () => logToolCall(logger, "generate_uuid", tools.generateUUID)(),
  {
    name: "generate_uuid",
    description: "Generate a unique UUID identifier.",
    schema: z.object({})
  }
);

const AGENT_TOOLS = [calculatorTool, resolveDatetimeTool, generateUuidTool];

export class LangChainAgentManager {
  framework = "LangChain Agent";
  toolNames = ["calculator", "resolve_datetime", "generate_uuid"];
  modelIdentifiers = ALL_MODEL_IDENTIFIERS;
  toolTriggerHelp =
    "Tools are selected automatically from your prompt; you do not need to type a tool name. " +
    "If you want a specific behavior, ask explicitly " +
    "(for example: 'calculate 20 * 5', 'parse tomorrow at 2pm', or 'generate a unique ticket ID').";

  constructor(model, stream = true) {
    const config = getIdentifierMappings()[model];
    this.provider = config?.provider ?? "unknown";
    this.model = config?.model ?? model;
    this.stream = stream;

    logger.info(
      `Initializing LangChain agent | provider=${this.provider} | model=${this.model} | stream=${this.stream}`
    );

    this.agent = createAgent({
      model: `${this.provider}:${this.model}`,
      tools: AGENT_TOOLS,
      systemPrompt:
        "You are an AI assistant that can use tools. " +
        "Use the calculator for arithmetic, resolve_datetime for date/time phrases, " +
        "and generate_uuid when the user asks for a unique ID, UUID, ticket ID, or identifier. " +
        "Think step-by-step, use tools when needed, and return a concise final answer."
    });
  }

  async askQuestion(topic) {
    try {
      const localToolCall = tools.routeToolForPrompt(topic, this.toolNames);
      if (localToolCall) {
        logger.info(`Processing prompt locally | tool=${localToolCall.name} | chars=${topic.length}`);
        const observation = tools.runTool(localToolCall.name, localToolCall.input);
        return {
          success: true,
          stream: false,
          provider: this.provider,
          model: this.model,
          prompt: topic,
          localOnly: true,
          selectedTool: localToolCall.name,
          response: JSON.stringify(observation, null, 2)
        };
      }

      logger.info(`Processing prompt with LLM | chars=${topic.length}`);
      const input = { messages: [{ role: "user", content: topic }] };

      const result = this.stream
        ? await this.agent.stream(input, { streamMode: ["messages", "updates"] })
        : await this.agent.invoke(input);

      return {
        success: true,
        stream: this.stream,
        provider: this.provider,
        model: this.model,
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
  const startupModel = await selectStartupModel(ALL_MODEL_IDENTIFIERS, args.mode, args.modelIdentifier);
  const manager = new LangChainAgentManager(startupModel, args.stream);
  await runMode(manager, args.mode, args.host, args.port, args.stream);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
