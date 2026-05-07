#!/usr/bin/env node

import { createAgent } from 'langchain';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

import * as tools from '../chapter_1/tools.js';
import { ALL_MODEL_IDENTIFIERS, buildCommonArgs, extractTextContent, getChapterLogger, getIdentifierMappings, langChainMessageToolCalls, langChainMessageTypeName, langChainStreamChunkFromEvent, runMode, selectStartupModel } from '../chapter_1/utils.js';

const logger = getChapterLogger('volume_2.chapter_2.agent_basic');
const SYSTEM_PROMPT = 'Use generate_uuid when user asks for UUID. Keep responses short.';

function printStepHeader(step, typeName) {
  console.log(`\n[${step}] ${typeName}`);
}

function printStepMessage(step, message, content = extractTextContent(message?.content)) {
  printStepHeader(step, langChainMessageTypeName(message));
  console.log(content);
}

function flushPendingToolLogs(state) {
  while (state.pendingToolLogs.length) {
    const { name, input, output } = state.pendingToolLogs.shift();
    logger.info(`Tool call | name=${name} | input=%o`, input);
    logger.info(`Tool result | name=${name} | output=%o`, output);
  }
}

function printBasicAgentStepMessage(message, state, { stream = false } = {}) {
  if (!message) return;

  const typeName = langChainMessageTypeName(message);
  const toolCalls = langChainMessageToolCalls(message);
  if (typeName === 'SystemMessage') {
    if (!state.printedSystemMessage) {
      printStepMessage('STEP 1 - SYSTEM MESSAGE', message);
      state.printedSystemMessage = true;
    }
    return;
  }
  if (typeName === 'HumanMessage') {
    if (!state.printedHumanMessage) {
      printStepMessage('STEP 2 - USER -> LLM', message);
      state.printedHumanMessage = true;
    }
    return;
  }
  if (typeName.includes('AIMessage') && toolCalls) {
    printStepHeader('STEP 3 - LLM -> AGENT TOOL INSTRUCTIONS', 'AIMessage.tool_calls');
    console.log(JSON.stringify(toolCalls, null, 2));
    return;
  }
  if (typeName === 'ToolMessage') {
    flushPendingToolLogs(state);
    printStepMessage('STEP 4 - TOOL -> LLM', message);
    return;
  }
  if (typeName === 'AIMessageChunk') {
    const delta = extractTextContent(message.content);
    if (delta && !state.printedFinalHeader) {
      printStepHeader('STEP 5 - LLM FINAL MESSAGE', 'AIMessage');
      state.printedFinalHeader = true;
    }
    if (delta) process.stdout.write(delta);
    state.finalText += delta;
    return;
  }
  if (typeName.includes('AIMessage')) {
    const text = extractTextContent(message.content);
    if (stream) {
      if (!state.printedFinalHeader) {
        printStepMessage('STEP 5 - LLM FINAL MESSAGE', message, text);
        state.printedFinalHeader = true;
      }
      if (!state.finalText) state.finalText = text;
    } else {
      printStepMessage('STEP 5 - LLM FINAL MESSAGE', message, text);
      state.printedFinalHeader = true;
      state.finalText = text;
    }
  }
}

function createBasicAgentStepState(pendingToolLogs = []) {
  return { finalText: '', pendingToolLogs, printedFinalHeader: false, printedSystemMessage: false, printedHumanMessage: false };
}

async function printBasicAgentStepOutput({ messages = [], streamEvents = null, state = createBasicAgentStepState() } = {}) {
  for (const message of messages) printBasicAgentStepMessage(message, state);
  if (streamEvents) {
    for await (const event of streamEvents) {
      printBasicAgentStepMessage(langChainStreamChunkFromEvent(event), state, { stream: true });
    }
  }
  console.log('\n');
  return state.finalText.trim();
}

export class LangChainUuidAgentManager {
  framework = 'LangChain Basic Agent';
  printsOwnOutput = true;
  toolNames = ['generate_uuid'];
  toolTriggerHelp = "Tools are triggered automatically. Ask for a UUID/ticket ID to trigger generate_uuid.";
  modelIdentifiers = ALL_MODEL_IDENTIFIERS;

  constructor(model) {
    const config = getIdentifierMappings()[model];
    this.provider = config?.provider ?? 'unknown';
    this.model = config?.model ?? model;
    const providerName = this.provider === 'google' ? 'google-genai' : this.provider;
    this.pendingToolLogs = [];
    const generateUuidTool = tool((input) => {
      const output = tools.generateUUID(input);
      this.pendingToolLogs.push({ name: 'generate_uuid', input, output });
      return output;
    }, {
      name: 'generate_uuid',
      description: 'Generate a unique UUID identifier.',
      schema: z.object({}),
    });
    this.agent = createAgent({
      model: `${providerName}:${this.model}`,
      tools: [generateUuidTool],
      systemPrompt: SYSTEM_PROMPT,
    });
  }

  async askQuestion(topic, options = {}) {
    try {
      const initialMessages = [new SystemMessage(SYSTEM_PROMPT), new HumanMessage(topic)];
      const shouldStream = Boolean(options.stream);
      this.pendingToolLogs.length = 0;
      const state = createBasicAgentStepState(this.pendingToolLogs);
      for (const message of initialMessages) printBasicAgentStepMessage(message, state);
      let finalText;

      if (shouldStream) {
        const stream = await this.agent.stream({ messages: [initialMessages[1]] }, { streamMode: ['messages'] });
        finalText = await printBasicAgentStepOutput({ streamEvents: stream, state });
      } else {
        const response = await this.agent.invoke({ messages: [initialMessages[1]] });
        const responseMessages = Array.isArray(response?.messages) ? response.messages : [];
        finalText = await printBasicAgentStepOutput({ messages: responseMessages, state });
      }

      return { success: true, finalText };
    } catch (error) {
      logger.error(error);
      return { success: false, error: error?.message || String(error) };
    }
  }
}

async function main() {
  const args = buildCommonArgs();
  const startupModel = await selectStartupModel(ALL_MODEL_IDENTIFIERS, args.mode, args.modelIdentifier);
  const manager = new LangChainUuidAgentManager(startupModel);
  await runMode(manager, args.mode, args.host, args.port, args.stream);
}

if (import.meta.url === `file://${process.argv[1]}`) main();
