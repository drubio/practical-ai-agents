#!/usr/bin/env node

import { createAgent } from 'langchain';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

import * as tools from '../chapter_1/tools.js';
import {
  ALL_MODEL_IDENTIFIERS,
  buildCommonArgs,
  defaultChunkIterator,
  getChapterLogger,
  getIdentifierMappings,
  logToolCall,
  runMode,
  selectStartupModel,
} from '../chapter_1/utils.js';

const logger = getChapterLogger('volume_2.chapter_2.agent_uuid');

const generateUuidTool = tool(
  () => logToolCall(logger, 'generate_uuid', tools.generateUUID)(),
  {
    name: 'generate_uuid',
    description: 'Generate a unique UUID identifier.',
    schema: z.object({}),
  }
);

export class LangChainUuidAgentManager {
  framework = 'LangChain UUID Agent';
  toolNames = ['generate_uuid'];
  modelIdentifiers = ALL_MODEL_IDENTIFIERS;

  constructor(model, stream = true) {
    const config = getIdentifierMappings()[model];
    this.activeModelIdentifier = model;
    this.provider = config?.provider ?? 'unknown';
    this.model = config?.model ?? model;
    this.stream = stream;

    const providerName = this.provider === 'google' ? 'google-genai' : this.provider;
    const selectedModel = providerName && providerName !== 'unknown'
      ? `${providerName}:${this.model}`
      : this.model;

    this.agent = createAgent({
      model: selectedModel,
      tools: [generateUuidTool],
      systemPrompt:
        'You are an AI assistant that can use only one tool: generate_uuid. ' +
        'Use generate_uuid whenever the user asks for a UUID or unique identifier.',
    });
  }

  async askQuestion(topic) {
    try {
      const input = { messages: [{ role: 'user', content: topic }] };
      const result = this.stream
        ? await this.agent.stream(input, { streamMode: ['messages', 'updates'] })
        : await this.agent.invoke(input);

      return {
        success: true,
        stream: this.stream,
        provider: this.provider,
        model: this.model,
        prompt: topic,
        response: result,
      };
    } catch (error) {
      logger.error('LangChain UUID askQuestion failed', error);
      return {
        success: false,
        stream: this.stream,
        provider: this.provider,
        model: this.model,
        prompt: topic,
        error: error?.message || String(error),
        response: null,
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
  const manager = new LangChainUuidAgentManager(startupModel, args.stream);
  await runMode(manager, args.mode, args.host, args.port, args.stream);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
