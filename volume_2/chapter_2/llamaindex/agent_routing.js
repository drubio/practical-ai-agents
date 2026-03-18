#!/usr/bin/env node

import * as tools from "../../chapter_1/tools.js";
import {
  buildTaskPrompt,
  buildCommonArgs,
  defaultChunkIterator,
  extractOutputText,
  getChapterLogger,
  logToolCall,
  describeModelAvailability,
  getRoutableModelIdentifiers,
  runMode
} from "../../chapter_1/utils.js";
import { ALL_MODEL_IDENTIFIERS, createLlamaindexLLM, routeModelForPrompt } from "../../chapter_1/models.js";

const logger = getChapterLogger("volume_2.chapter_2.llamaindex.agent_routing");
const FINAL_RESPONSE_INSTRUCTION = [
  "Treat each non-empty line in the user prompt as a required task and handle every requested step before answering.",
  "You must call calculator for arithmetic expressions or fee calculations, resolve_datetime for scheduling/date phrases, and generate_uuid for unique IDs.",
  'Return your final answer as JSON with this shape: {"text": "<human readable summary>", "raw": {"ticket_id": null, "meeting": null, "calculations": []}}.',
  "Populate raw fields with structured values when available and use null or empty arrays when unavailable.",
  "If any line requests arithmetic, raw.calculations must include the calculator result."
].join(" ");

export const ALL_TOOL_NAMES = ["calculator", "resolve_datetime", "generate_uuid"];

export function buildToolMap(logToolCallFn, activeLogger) {
  return {
    calculator: logToolCallFn(activeLogger, "calculator", tools.calculator),
    resolve_datetime: logToolCallFn(activeLogger, "resolve_datetime", tools.resolveDatetime),
    generate_uuid: logToolCallFn(activeLogger, "generate_uuid", tools.generateUUID),
  };
}

export function selectToolMap(logToolCallFn, activeLogger, toolNames) {
  const available = buildToolMap(logToolCallFn, activeLogger);
  return Object.fromEntries(toolNames.map((name) => [name, available[name]]));
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
  modelIdentifiers = [];
  toolTriggerHelp =
    "Tools are selected automatically from your prompt; you do not need to type a tool name. If you want a specific behavior, ask explicitly (for example: 'calculate 20 * 5' or 'parse tomorrow at 2pm').";

  constructor(modelIdentifiers, initialModel, stream = false) {
    this.modelIdentifiers = modelIdentifiers;
    this.modelIdentifier = initialModel;
    this.provider = "unknown";
    this.model = initialModel;
    this.stream = stream;
    logger.info(`Initializing LlamaIndex routing agent | provider=${this.provider} | model=${this.model} | stream=${this.stream}`);
    this.llmCache = new Map();
  }

  getLlmByIdentifier(modelIdentifier) {
    if (!this.llmCache.has(modelIdentifier)) {
      this.llmCache.set(modelIdentifier, createLlamaindexLLM(modelIdentifier));
    }
    return this.llmCache.get(modelIdentifier);
  }

  async askQuestion(topic) {
    try {
      logger.info(`Received prompt | chars=${topic.length} | multiline=${topic.includes("\n")}`);
      logger.info("Delegating full prompt to routed LlamaIndex agent");
      const selectedToolNames = ALL_TOOL_NAMES;
      const selectedModel = routeModelForPrompt(topic, selectedToolNames, this.modelIdentifiers);
      this.modelIdentifier = selectedModel.name;

      const resolved = this.getLlmByIdentifier(selectedModel.name);
      this.provider = resolved.provider;
      this.model = resolved.model;

      const toolMap = selectToolMap(logToolCall, logger, selectedToolNames);

      const messages = [
        {
          role: "system",
          content: [
            "You are an AI assistant that can use tools.",
            "Choose the best tool(s) among those provided.",
            FINAL_RESPONSE_INSTRUCTION,
            "When needed, reply strictly in this format and nothing else:",
            "TOOL: <tool_name>",
            "INPUT: <valid JSON or plain text>",
            "If no tool is needed, return a concise final answer directly."
          ].join("\n")
        },
        { role: "user", content: buildTaskPrompt(topic) }
      ];

      logger.info("Awaiting LlamaIndex agent response");
      for (let attempt = 0; attempt < 6; attempt += 1) {
        const result = await resolved.llm.chat({ messages });
        const text = extractOutputText(result).trim();
        const toolCall = parseToolCall(text, toolMap);

        if (!toolCall) {
          return {
            success: true,
            stream: this.stream,
            provider: resolved.provider,
            model: resolved.model,
            modelName: selectedModel.name,
            modelTier: selectedModel.tier,
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
  const modelIdentifiers = await getRoutableModelIdentifiers(ALL_MODEL_IDENTIFIERS, args.modelIdentifier);
  const manager = new LlamaIndexAgentRoutingManager(modelIdentifiers, modelIdentifiers[0], args.stream);
  manager.toolTriggerHelp = `${manager.toolTriggerHelp} ${await describeModelAvailability(ALL_MODEL_IDENTIFIERS)}`;
  await runMode(manager, args.mode, args.host, args.port, args.stream);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
