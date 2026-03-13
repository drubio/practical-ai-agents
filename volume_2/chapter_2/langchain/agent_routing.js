#!/usr/bin/env node

import * as z from "../../chapter_1/node_modules/zod/index.js";
import { createAgent, tool } from "../../chapter_1/node_modules/langchain/dist/index.js";

import * as tools from "../../chapter_1/tools.js";
import {
  buildCommonArgs,
  defaultChunkIterator,
  getChapterLogger,
  logToolCall,
  selectStartupModel,
  runMode
} from "../../chapter_1/utils.js";
import { ALL_MODEL_IDENTIFIERS, getIdentifierMappings } from "../../chapter_1/models.js";

const logger = getChapterLogger("volume_2.chapter_2.langchain.agent_routing");

export const ALL_TOOL_NAMES = [
  "summarize_text",
  "extract_keywords",
  "extract_tasks",
  "score_priority",
  "route_workflow",
  "parse_content",
  "resolve_datetime",
  "format_json",
  "calculator",
  "analyze_text"
];

export function buildTools(logToolCallFn, activeLogger) {
  return {
    summarize_text: tool(logToolCallFn(activeLogger, "summarize_text", ({ text }) => tools.summarizeText(text)), {
      name: "summarize_text",
      description: "Summarize text.",
      schema: z.object({ text: z.string() })
    }),
    extract_keywords: tool(logToolCallFn(activeLogger, "extract_keywords", ({ text }) => tools.extractKeywords(text)), {
      name: "extract_keywords",
      description: "Extract keywords.",
      schema: z.object({ text: z.string() })
    }),
    extract_tasks: tool(logToolCallFn(activeLogger, "extract_tasks", ({ text }) => tools.extractTasks(text)), {
      name: "extract_tasks",
      description: "Extract tasks from text.",
      schema: z.object({ text: z.string() })
    }),
    score_priority: tool(logToolCallFn(activeLogger, "score_priority", ({ text }) => tools.scorePriority(text)), {
      name: "score_priority",
      description: "Score priority from text.",
      schema: z.object({ text: z.string() })
    }),
    route_workflow: tool(logToolCallFn(activeLogger, "route_workflow", ({ text }) => tools.routeWorkflow(text)), {
      name: "route_workflow",
      description: "Route workflow from text.",
      schema: z.object({ text: z.string() })
    }),
    parse_content: tool(logToolCallFn(activeLogger, "parse_content", ({ content }) => tools.parseContent(content)), {
      name: "parse_content",
      description: "Parse content.",
      schema: z.object({ content: z.string() })
    }),
    resolve_datetime: tool(logToolCallFn(activeLogger, "resolve_datetime", ({ text }) => tools.resolveDatetime(text)), {
      name: "resolve_datetime",
      description: "Resolve datetime from text.",
      schema: z.object({ text: z.string() })
    }),
    format_json: tool(logToolCallFn(activeLogger, "format_json", ({ input }) => tools.formatJson(input)), {
      name: "format_json",
      description: "Format JSON-like input.",
      schema: z.object({ input: z.any() })
    }),
    calculator: tool(logToolCallFn(activeLogger, "calculator", ({ expression }) => tools.calculator(expression)), {
      name: "calculator",
      description: "Evaluate expression.",
      schema: z.object({ expression: z.string() })
    }),
    analyze_text: tool(logToolCallFn(activeLogger, "analyze_text", ({ text }) => tools.analyzeText(text)), {
      name: "analyze_text",
      description: "Analyze text.",
      schema: z.object({ text: z.string() })
    })
  };
}

export function selectTools(logToolCallFn, activeLogger, toolNames) {
  const available = buildTools(logToolCallFn, activeLogger);
  return toolNames.map((name) => available[name]);
}

function extractOutput(result) {
  if (typeof result?.output === "string") return result.output;
  const messages = result?.messages ?? [];
  const content = messages[messages.length - 1]?.content;
  if (typeof content === "string") return content;
  return String(result ?? "");
}

export class LangChainAgentRoutingManager {
  framework = "LangChain Agent Routing";
  toolNames = ALL_TOOL_NAMES;
  modelIdentifiers = ALL_MODEL_IDENTIFIERS;
  toolTriggerHelp =
    "Tools are selected automatically from your prompt; you do not need to type a tool name. If you want a specific behavior, ask explicitly (for example: 'extract tasks and score priority').";

  constructor(model) {
    const config = getIdentifierMappings()[model];
    this.provider = config?.provider ?? "unknown";
    this.model = config?.model ?? model;
    logger.info(`Initializing LangChain routing agent | provider=${this.provider} | model=${this.model}`);
    this.agent = createAgent({
      model: this.model,
      tools: selectTools(logToolCall, logger, ALL_TOOL_NAMES),
      systemPrompt:
        "You are an AI assistant that can use tools. Think step-by-step, use tools when needed, and return a concise final answer."
    });
  }

  async askQuestion(topic) {
    try {
      logger.info(`Processing prompt | chars=${topic.length}`);
      const result = await this.agent.invoke({ messages: [{ role: "user", content: topic }] });
      return { success: true, provider: this.provider, model: this.model, prompt: topic, response: extractOutput(result) };
    } catch (error) {
      logger.error("LangChain askQuestion failed", error);
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
  const manager = new LangChainAgentRoutingManager(startupModel);
  await runMode(manager, args.mode, args.host, args.port, args.stream);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
