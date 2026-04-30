#!/usr/bin/env node

import { createAgent } from 'langchain';
import { HumanMessage, AIMessage, ToolMessage, SystemMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

import * as tools from '../chapter_1/tools.js';
import { ALL_MODEL_IDENTIFIERS, buildCommonArgs, getChapterLogger, getIdentifierMappings, logToolCall, runMode, selectStartupModel } from '../chapter_1/utils.js';

const logger = getChapterLogger('volume_2.chapter_2.agent_basic');
const SYSTEM_PROMPT = 'Use generate_uuid when user asks for UUID. Keep responses short.';

const generateUuidTool = tool(() => logToolCall(logger, 'generate_uuid', tools.generateUUID)(), {
  name: 'generate_uuid',
  description: 'Generate a unique UUID identifier.',
  schema: z.object({}),
});

function textFromContent(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) return content.map((c) => (typeof c === 'string' ? c : (c?.text || c?.content || ''))).join('');
  if (content && typeof content === 'object') return content.text || content.content || '';
  return '';
}

function printMsg(tag, msg) {
  console.log(`\n[${tag}] ${msg.constructor?.name || 'Message'}`);
  console.log(textFromContent(msg.content));
}

export class LangChainUuidAgentManager {
  framework = 'LangChain Basic Agent';
  toolNames = ['generate_uuid'];
  toolTriggerHelp = "Tools are triggered automatically. Ask for a UUID/ticket ID to trigger generate_uuid.";
  modelIdentifiers = ALL_MODEL_IDENTIFIERS;

  constructor(model) {
    const config = getIdentifierMappings()[model];
    this.provider = config?.provider ?? 'unknown';
    this.model = config?.model ?? model;
    const providerName = this.provider === 'google' ? 'google-genai' : this.provider;
    this.agent = createAgent({
      model: `${providerName}:${this.model}`,
      tools: [generateUuidTool],
      systemPrompt: SYSTEM_PROMPT,
    });
  }

  async askQuestion(topic, options = {}) {
    try {
      printMsg('STEP 1 - SYSTEM MESSAGE', new SystemMessage(SYSTEM_PROMPT));
      const human = new HumanMessage(topic);
      printMsg('STEP 2 - USER -> LLM', human);

      const shouldStream = Boolean(options.stream);
      let finalText = '';

      if (shouldStream) {
        const stream = await this.agent.stream({ messages: [human] }, { streamMode: ['messages'] });

        for await (const event of stream) {
        const chunk = Array.isArray(event) ? event[1]?.[0] ?? event[1] : event;
        if (!chunk) continue;

        const type = chunk.constructor?.name || '';
        if (type.includes('AIMessage') && Array.isArray(chunk.tool_calls) && chunk.tool_calls.length) {
          console.log('\n[STEP 3 - LLM -> AGENT TOOL INSTRUCTIONS] AIMessage.tool_calls');
          console.log(JSON.stringify(chunk.tool_calls, null, 2));
        }

        if (type === 'AIMessageChunk') {
          const delta = textFromContent(chunk.content);
          if (delta) process.stdout.write(delta);
          finalText += delta;
        } else if (type === 'ToolMessage') {
          printMsg('STEP 4 - TOOL -> LLM', new ToolMessage(textFromContent(chunk.content), chunk.tool_call_id));
        } else if (type.includes('AIMessage')) {
          printMsg('STEP 5 - LLM FINAL MESSAGE', new AIMessage(textFromContent(chunk.content)));
        }
        }
      } else {
        const response = await this.agent.invoke({ messages: [human] });
        const messages = Array.isArray(response?.messages) ? response.messages : [];

        for (const message of messages) {
          const type = message?.constructor?.name || '';
          if (type.includes('AIMessage') && Array.isArray(message.tool_calls) && message.tool_calls.length) {
            console.log('\n[STEP 3 - LLM -> AGENT TOOL INSTRUCTIONS] AIMessage.tool_calls');
            console.log(JSON.stringify(message.tool_calls, null, 2));
          }

          if (type === 'ToolMessage') {
            printMsg('STEP 4 - TOOL -> LLM', new ToolMessage(textFromContent(message.content), message.tool_call_id));
          } else if (type.includes('AIMessage')) {
            const text = textFromContent(message.content);
            printMsg('STEP 5 - LLM FINAL MESSAGE', new AIMessage(text));
            finalText = text;
          }
        }
      }

      console.log('\n');
      return { success: true, finalText: finalText.trim() };
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
