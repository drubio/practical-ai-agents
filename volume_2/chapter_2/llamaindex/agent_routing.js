#!/usr/bin/env node

import { OpenAI } from "../../chapter_1/node_modules/@llamaindex/openai/dist/index.js";

import * as tools from "../../chapter_1/tools.js";
import {
  buildCommonArgs,
  defaultChunkIterator,
  getChapterLogger,
  logToolCall,
  selectStartupModel,
  runMode
} from "../../chapter_1/utils.js";
import { CHAPTER_1_MODEL_NAMES } from "../../chapter_1/models.js";

const logger = getChapterLogger("volume_2.chapter_2.llamaindex.agent_routing");

export const CHAPTER_1_TOOL_NAMES = ["summarize_text"];

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

export function buildToolMap(logToolCallFn, activeLogger) {
  return {
    summarize_text: logToolCallFn(activeLogger, "summarize_text", tools.summarizeText),
    extract_keywords: logToolCallFn(activeLogger, "extract_keywords", tools.extractKeywords),
    extract_tasks: logToolCallFn(activeLogger, "extract_tasks", tools.extractTasks),
    score_priority: logToolCallFn(activeLogger, "score_priority", tools.scorePriority),
    route_workflow: logToolCallFn(activeLogger, "route_workflow", tools.routeWorkflow),
    parse_content: logToolCallFn(activeLogger, "parse_content", tools.parseContent),
    resolve_datetime: logToolCallFn(activeLogger, "resolve_datetime", tools.resolveDatetime),
    format_json: logToolCallFn(activeLogger, "format_json", tools.formatJson),
    calculator: logToolCallFn(activeLogger, "calculator", tools.calculator),
    analyze_text: logToolCallFn(activeLogger, "analyze_text", tools.analyzeText)
  };
}

export function selectToolMap(logToolCallFn, activeLogger, toolNames) {
  const available = buildToolMap(logToolCallFn, activeLogger);
  return Object.fromEntries(toolNames.map((name) => [name, available[name]]));
}

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

function parseToolCall(text, activeToolMap) {
  const match = text.match(/TOOL:\s*([a-z_]+)\s*\nINPUT:\s*([\s\S]*)$/i);
  if (!match) return null;

  const name = match[1].trim();
  const rawInput = match[2].trim();
  if (!(name in activeToolMap)) return null;

  try {
    return { name, input: JSON.parse(rawInput) };
  } catch {
    return { name, input: rawInput };
  }
}

export class LlamaIndexAgentRoutingManager {
  framework = "LlamaIndex Agent Routing";
  toolNames = ALL_TOOL_NAMES;
  modelNames = CHAPTER_1_MODEL_NAMES;
  toolTriggerHelp =
    "Tools are selected automatically from your prompt; you do not need to type a tool name. If you want a specific behavior, ask explicitly (for example: 'extract tasks and score priority').";

  constructor(model = "gpt-5.2") {
    this.model = model;
    logger.info(`Initializing LlamaIndex routing agent | model=${model}`);
    this.llm = new OpenAI({ model });
    this.toolMap = selectToolMap(logToolCall, logger, ALL_TOOL_NAMES);
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

      for (let attempt = 0; attempt < 6; attempt += 1) {
        const result = await this.llm.chat({ messages });
        const text = extractText(result).trim();
        const toolCall = parseToolCall(text, this.toolMap);

        if (!toolCall) {
          return { success: true, provider: "openai", model: this.model, prompt: topic, response: text };
        }

        const observation = this.toolMap[toolCall.name](toolCall.input);
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
  const startupModel = await selectStartupModel(CHAPTER_1_MODEL_NAMES, args.mode, args.model);
  const manager = new LlamaIndexAgentRoutingManager(startupModel);
  await runMode(manager, args.mode, args.host, args.port, args.stream);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
