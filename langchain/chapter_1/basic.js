#!/usr/bin/env node

import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { createAgent } from 'langchain';

import { createLangChainModel } from '../../shared/utils.mjs';
import { createGenerateUuidTool } from '../../shared/langchain/tools.js';
import {
  ALL_MODEL_IDENTIFIERS,
  buildCommonArgs,
  createAgentStepState,
  getChapterLogger,
  getIdentifierMappings,
  printAgentStepMessage,
  printAgentStepOutput,
  runMode,
  selectStartupModel,
} from '../../shared/langchain/utils.js';

const logger = getChapterLogger('langchain.chapter_1.basic');
const SYSTEM_PROMPT = 'Use generate_uuid when user asks for UUID. Keep responses short.';

export class LangChainAgentManager {
  framework = 'LangChain Basic Agent';
  printsOwnOutput = true;
  toolNames = ['generate_uuid'];
  toolTriggerHelp = 'Tools are triggered automatically. Ask for a UUID/ticket ID to trigger generate_uuid.';
  modelIdentifiers = ALL_MODEL_IDENTIFIERS;

  constructor(model, { temperature = 0.7, maxTokens = 1000 } = {}) {
    const modelConfig = getIdentifierMappings()[model];
    this.modelIdentifier = model;
    this.provider = modelConfig?.provider ?? 'openai';
    this.model = modelConfig?.model ?? model;
    this.temperature = temperature;
    this.maxTokens = maxTokens;
    this.pendingToolLogs = [];
    this.agent = this._buildAgent();
  }

  _createModel() {
    return createLangChainModel(this.modelIdentifier, {
      temperature: this.temperature,
      maxTokens: this.maxTokens,
    });
  }

  _buildTools() {
    return [createGenerateUuidTool(this.pendingToolLogs)];
  }

  _buildAgent() {
    return createAgent({
      model: this._createModel(),
      tools: this._buildTools(),
      systemPrompt: SYSTEM_PROMPT,
    });
  }

  _buildMessages(topic) {
    return [
      new SystemMessage(SYSTEM_PROMPT),
      new HumanMessage(topic),
    ];
  }

  async askQuestion(topic, options = {}) {
    try {
      const initialMessages = this._buildMessages(topic);
      const humanMessage = initialMessages[1];

      this.pendingToolLogs.length = 0;
      const state = createAgentStepState(this.pendingToolLogs);
      for (const message of initialMessages) printAgentStepMessage(message, state, logger);

      let finalText;
      if (options.stream) {
        const stream = await this.agent.stream({ messages: [humanMessage] }, { streamMode: ['messages'] });
        finalText = await printAgentStepOutput({ logger, streamEvents: stream, state });
      } else {
        const response = await this.agent.invoke({ messages: [humanMessage] });
        const responseMessages = Array.isArray(response?.messages) ? response.messages : [];
        finalText = await printAgentStepOutput({ logger, messages: responseMessages, state });
      }

      return {
        success: true,
        finalText,
        final_text: finalText,
        provider: this.provider,
        model: this.model,
        temperature: this.temperature,
        maxTokens: this.maxTokens,
      };
    } catch (error) {
      logger.error(error);
      return { success: false, error: error?.message || String(error) };
    }
  }
}

async function main() {
  const args = buildCommonArgs();
  const startupModel = await selectStartupModel(ALL_MODEL_IDENTIFIERS, args.mode, args.modelIdentifier);
  const manager = new LangChainAgentManager(startupModel, { temperature: args.temperature, maxTokens: args.maxTokens });
  await runMode(manager, args.mode, args.host, args.port, args.stream);
}

if (import.meta.url === `file://${process.argv[1]}`) main();
