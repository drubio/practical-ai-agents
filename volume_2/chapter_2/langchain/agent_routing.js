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

function escapeRegex(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function triggerMatch(promptLower, trigger) {
  if (trigger.includes(" ") || [":", "/", ".", "-"].some((ch) => trigger.includes(ch))) {
    return promptLower.includes(trigger);
  }
  return new RegExp(`\\b${escapeRegex(trigger)}\\b`).test(promptLower);
}

export function routeToolsForPrompt(prompt) {
  const promptLower = prompt.toLowerCase();
  const selected = new Set();

  const keywordRoutes = {
    summarize_text: ["summarize", "tl;dr", "overview", "recap"],
    extract_keywords: ["keyword", "key phrase", "tags", "topics"],
    extract_tasks: ["todo", "task", "action item", "next steps"],
    score_priority: ["priority", "urgent", "severity", "p0", "p1"],
    route_workflow: ["workflow", "route", "triage", "handoff"],
    parse_content: ["parse", "extract fields", "structured", "html"],
    resolve_datetime: ["date", "time", "schedule", "tomorrow", "next week"],
    format_json: ["json", "yaml", "format", "schema"],
    calculator: ["calculate", "math", "equation", "percentage"],
    analyze_text: ["analyze", "analysis", "sentiment", "tone", "readability"]
  };

  for (const [toolName, triggers] of Object.entries(keywordRoutes)) {
    if (triggers.some((trigger) => triggerMatch(promptLower, trigger))) {
      selected.add(toolName);
    }
  }

  const hasMathExpression = ["+", "*", "/", "="].some((op) => prompt.includes(op)) || prompt.includes(" - ");
  if (hasMathExpression) {
    selected.add("calculator");
  }

  if (!selected.size) return ALL_TOOL_NAMES;

  if (selected.has("extract_tasks") && !selected.has("score_priority")) {
    selected.add("score_priority");
  }
  if (selected.has("route_workflow") && !selected.has("extract_tasks")) {
    selected.add("extract_tasks");
  }

  return ALL_TOOL_NAMES.filter((name) => selected.has(name));
}

export class LangChainAgentRoutingManager {
  framework = "LangChain Agent Routing";
  toolNames = ALL_TOOL_NAMES;
  modelIdentifiers = ALL_MODEL_IDENTIFIERS;
  toolTriggerHelp =
    "Tools are selected automatically from your prompt; you do not need to type a tool name. If you want a specific behavior, ask explicitly (for example: 'extract tasks and score priority').";

  constructor(model, stream = true) {
    const config = getIdentifierMappings()[model];
    this.provider = config?.provider ?? "unknown";
    this.model = config?.model ?? model;
    this.stream = stream;
    this.agentCache = new Map();
    logger.info(
      `Initializing LangChain routing agent | provider=${this.provider} | initial_model=${this.model} | stream=${this.stream}`
    );
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
            "Choose the best tool(s) among those provided, then return a concise final answer."
        })
      );
    }
    return this.agentCache.get(key);
  }

  async askQuestion(topic) {
    try {
      logger.info(`Processing prompt | chars=${topic.length}`);
      const selectedToolNames = routeToolsForPrompt(topic);
      const agent = this.getAgent(this.provider, this.model, selectedToolNames);
      const input = { messages: [{ role: "user", content: topic }] };
      const result = this.stream
        ? await agent.stream(input, { streamMode: ["messages", "updates"] })
        : await agent.invoke(input);

      return {
        success: true,
        stream: this.stream,
        provider: this.provider,
        model: this.model,
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
  const startupModel = await selectStartupModel(ALL_MODEL_IDENTIFIERS, args.mode, args.modelIdentifier);
  const manager = new LangChainAgentRoutingManager(startupModel, args.stream);
  await runMode(manager, args.mode, args.host, args.port, args.stream);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
