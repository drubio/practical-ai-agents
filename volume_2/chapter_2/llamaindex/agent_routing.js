#!/usr/bin/env node

import * as tools from "../../chapter_1/tools.js";
import {
  buildCommonArgs,
  defaultChunkIterator,
  extractOutputText,
  getChapterLogger,
  logToolCall,
  selectStartupModel,
  runMode
} from "../../chapter_1/utils.js";
import { ALL_MODEL_IDENTIFIERS, createLlamaindexLLM } from "../../chapter_1/models.js";

const logger = getChapterLogger("volume_2.chapter_2.llamaindex.agent_routing");

export const ALL_TOOL_NAMES = ["calculator", "resolve_datetime", "format_json"];

export function buildToolMap(logToolCallFn, activeLogger) {
  return {
    calculator: logToolCallFn(activeLogger, "calculator", tools.calculator),
    resolve_datetime: logToolCallFn(activeLogger, "resolve_datetime", tools.resolveDatetime),
    format_json: logToolCallFn(activeLogger, "format_json", tools.formatJson)
  };
}

export function selectToolMap(logToolCallFn, activeLogger, toolNames) {
  const available = buildToolMap(logToolCallFn, activeLogger);
  return Object.fromEntries(toolNames.map((name) => [name, available[name]]));
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
    calculator: ["calculate", "math", "equation", "percentage"],
    resolve_datetime: ["date", "time", "schedule", "tomorrow", "next week"],
    format_json: ["json", "yaml", "format", "schema"]
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

  return ALL_TOOL_NAMES.filter((name) => selected.has(name));
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
  modelIdentifiers = ALL_MODEL_IDENTIFIERS;
  toolTriggerHelp =
    "Tools are selected automatically from your prompt; you do not need to type a tool name. If you want a specific behavior, ask explicitly (for example: 'calculate 20 * 5 or format this JSON').";

  constructor(model, stream = false) {
    const { provider, model: resolvedModel, llm } = createLlamaindexLLM(model);
    this.provider = provider;
    this.model = resolvedModel;
    this.stream = stream;
    logger.info(`Initializing LlamaIndex routing agent | provider=${this.provider} | model=${this.model} | stream=${this.stream}`);
    this.llm = llm;
  }

  async askQuestion(topic) {
    try {
      logger.info(`Processing prompt | chars=${topic.length}`);
      const selectedToolNames = routeToolsForPrompt(topic);
      const toolMap = selectToolMap(logToolCall, logger, selectedToolNames);

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
        const text = extractOutputText(result).trim();
        const toolCall = parseToolCall(text, toolMap);

        if (!toolCall) {
          return {
            success: true,
            stream: this.stream,
            provider: this.provider,
            model: this.model,
            selectedTools: selectedToolNames,
            prompt: topic,
            response: text
          };
        }

        const observation = toolMap[toolCall.name](toolCall.input);
        messages.push({ role: "assistant", content: text });
        messages.push({ role: "user", content: `Tool result for ${toolCall.name}: ${JSON.stringify(observation)}` });
      }

      return {
        success: false,
        stream: this.stream,
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
  const manager = new LlamaIndexAgentRoutingManager(startupModel, args.stream);
  await runMode(manager, args.mode, args.host, args.port, args.stream);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
