#!/usr/bin/env node

import * as z from "zod";
import { createAgent, tool } from "langchain";

import {
  analyzeText,
  calculator,
  extractKeywords,
  extractTasks,
  formatJson,
  parseContent,
  resolveDatetime,
  routeWorkflow,
  scorePriority,
  summarizeText
} from "../tools.js";

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
    }),
    tool(logToolCall(logger, "extract_keywords", ({ text }) => extractKeywords(text)), {
      name: "extract_keywords",
      description: "Extract keywords.",
      schema: z.object({ text: z.string() })
    }),
    tool(logToolCall(logger, "extract_tasks", ({ text }) => extractTasks(text)), {
      name: "extract_tasks",
      description: "Extract tasks from text.",
      schema: z.object({ text: z.string() })
    }),
    tool(logToolCall(logger, "score_priority", ({ text }) => scorePriority(text)), {
      name: "score_priority",
      description: "Score priority from text.",
      schema: z.object({ text: z.string() })
    }),
    tool(logToolCall(logger, "route_workflow", ({ text }) => routeWorkflow(text)), {
      name: "route_workflow",
      description: "Route workflow from text.",
      schema: z.object({ text: z.string() })
    }),
    tool(logToolCall(logger, "parse_content", ({ content }) => parseContent(content)), {
      name: "parse_content",
      description: "Parse content.",
      schema: z.object({ content: z.string() })
    }),
    tool(logToolCall(logger, "resolve_datetime", ({ text }) => resolveDatetime(text)), {
      name: "resolve_datetime",
      description: "Resolve datetime from text.",
      schema: z.object({ text: z.string() })
    }),
    tool(logToolCall(logger, "format_json", ({ input }) => formatJson(input)), {
      name: "format_json",
      description: "Format JSON-like input.",
      schema: z.object({ input: z.any() })
    }),
    tool(logToolCall(logger, "calculator", ({ expression }) => calculator(expression)), {
      name: "calculator",
      description: "Evaluate expression.",
      schema: z.object({ expression: z.string() })
    }),
    tool(logToolCall(logger, "analyze_text", ({ text }) => analyzeText(text)), {
      name: "analyze_text",
      description: "Analyze text.",
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
  toolNames = [
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
