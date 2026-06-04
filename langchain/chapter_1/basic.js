#!/usr/bin/env node

import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { createAgent } from 'langchain';

import {
  BaseLLMManager,
  createLangChainModel,
  interactiveCli,
  printCliHelp,
} from '../../shared/utils.mjs';
import { createGenerateUuidTool } from '../../shared/langchain/tools.js';
import {
  createAgentStepState,
  extractTextContent,
  getChapterLogger,
  langChainStreamChunkFromEvent,
  printAgentStepMessage,
  printAgentStepOutput,
} from '../../shared/langchain/utils.js';

const logger = getChapterLogger('langchain.chapter_1.basic');
const SYSTEM_PROMPT = 'Use generate_uuid when user asks for UUID. Keep responses short.';

export class LangChainLLMManager extends BaseLLMManager {
  printsOwnOutput = true;
  toolNames = ['generate_uuid'];
  toolTriggerHelp = 'Tools are triggered automatically. Ask for a UUID/ticket ID to trigger generate_uuid.';

  constructor({ logStepByStep = true, stream = false } = {}) {
    super('LangChain Basic Agent');
    this.logStepByStep = logStepByStep;
    this.stream = stream;
    this.printsOwnOutput = logStepByStep || stream;
    this.pendingToolLogs = [];
  }

  async _testProvider(provider) {
    await this._createModel(this.providerModelIdentifier(provider), 0.7, 1000);
  }

  _createModel(selectedModel, temperature, maxTokens) {
    return createLangChainModel(selectedModel, {
      temperature,
      maxTokens,
    });
  }

  _buildTools() {
    return [createGenerateUuidTool(this.pendingToolLogs)];
  }

  _buildAgent(selectedModel, temperature, maxTokens) {
    return createAgent({
      model: this._createModel(selectedModel, temperature, maxTokens),
      tools: this._buildTools(),
      systemPrompt: SYSTEM_PROMPT,
    });
  }

  _buildMessages(prompt) {
    return [
      new SystemMessage(SYSTEM_PROMPT),
      new HumanMessage(prompt),
    ];
  }

  _extractText(result) {
    return result;
  }

  async _streamPlain(agent, humanMessage) {
    const parts = [];
    const stream = await agent.stream({ messages: [humanMessage] }, { streamMode: ['messages'] });
    for await (const event of stream) {
      const chunk = langChainStreamChunkFromEvent(event);
      const text = extractTextContent(chunk?.content ?? '');
      if (text) {
        process.stdout.write(text);
        parts.push(text);
      }
    }
    console.log();
    return parts.join('').trim();
  }

  _extractFinalText(responseMessages) {
    for (const message of [...responseMessages].reverse()) {
      const content = extractTextContent(message?.content ?? '');
      if (content.trim()) return content;
    }
    return '';
  }

  async askQuestion(topic, provider = null, template = '{topic}', maxTokens = 1000, temperature = 0.7) {
    const prompt = template.replace('{topic}', topic);
    const modelConfig = this.resolveModelConfig(provider);

    if (!modelConfig) {
      return {
        success: false,
        error: 'No providers available',
        provider: 'none',
        model: 'none',
        prompt,
        response: null,
      };
    }

    try {
      this.pendingToolLogs.length = 0;
      const initialMessages = this._buildMessages(prompt);
      const humanMessage = initialMessages[1];
      const agent = this._buildAgent(modelConfig.name, temperature, maxTokens);

      let finalText;
      if (this.logStepByStep) {
        const state = createAgentStepState(this.pendingToolLogs);
        for (const message of initialMessages) printAgentStepMessage(message, state, logger);

        if (this.stream) {
          const stream = await agent.stream({ messages: [humanMessage] }, { streamMode: ['messages'] });
          finalText = await printAgentStepOutput({ logger, streamEvents: stream, state });
        } else {
          const response = await agent.invoke({ messages: [humanMessage] });
          const responseMessages = Array.isArray(response?.messages) ? response.messages : [];
          finalText = await printAgentStepOutput({ logger, messages: responseMessages, state });
        }
      } else if (this.stream) {
        finalText = await this._streamPlain(agent, humanMessage);
      } else {
        const response = await agent.invoke({ messages: [humanMessage] });
        const responseMessages = Array.isArray(response?.messages) ? response.messages : [];
        finalText = this._extractFinalText(responseMessages);
      }

      return {
        success: true,
        provider: modelConfig.provider,
        model: modelConfig.model,
        modelIdentifier: modelConfig.name,
        prompt,
        response: this._extractText(finalText),
        finalText,
        final_text: finalText,
        temperature,
        maxTokens,
        logStepByStep: this.logStepByStep,
        log_step_by_step: this.logStepByStep,
        stream: this.stream,
      };
    } catch (error) {
      logger.error(error);
      return {
        success: false,
        provider: modelConfig.provider,
        model: modelConfig.model,
        modelIdentifier: modelConfig.name,
        prompt,
        error: error?.message || String(error),
        response: null,
        temperature,
        maxTokens,
        logStepByStep: this.logStepByStep,
        log_step_by_step: this.logStepByStep,
        stream: this.stream,
      };
    }
  }
}

async function main() {
  const args = process.argv.slice(2);
  if (args.includes('-h') || args.includes('--help')) {
    printCliHelp(process.argv[1], {
      description: 'Run the LangChain Chapter 1 basic tool-calling agent.',
      options: [
        ['--stream', 'Stream response tokens in CLI/web responses.'],
        ['--no-log-step-by-step', 'Hide LangChain message/tool step-by-step logs.'],
      ],
    });
    return;
  }
  const stream = args.includes('--stream');
  const logStepByStep = !args.includes('--no-log-step-by-step');

  if (args.includes('web')) {
    try {
      const { runWebServer } = await import('../../shared/essentials/web.mjs');
      await runWebServer(() => new LangChainLLMManager({ logStepByStep, stream }));
    } catch (error) {
      console.error('Error: shared web API not found or Express not installed.');
      console.error('Install Express: npm install express cors');
      process.exit(1);
    }
  } else {
    const manager = new LangChainLLMManager({ logStepByStep, stream });
    await manager._checkProviders();
    await interactiveCli(manager);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) main().catch(console.error);
