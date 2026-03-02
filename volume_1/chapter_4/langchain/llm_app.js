/**
 * LLM Tester - LangChain JavaScript Framework Implementation
 * Reusable core manager for chapter extensions.
 */

import { ChatAnthropic } from '@langchain/anthropic';
import { ChatOpenAI } from '@langchain/openai';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';

import {
    getApiKey,
    getDefaultModel,
    BaseLLMManager,
    interactiveCli,
} from '../utils.js';

class LangChainLLMManager extends BaseLLMManager {
    constructor() {
        super('LangChain JS');
    }

    async _testProvider(provider) {
        await this._createClient(provider, 0.7, 1000);
    }

    _createClient(provider, temperature, maxTokens) {
        if (provider === 'anthropic') {
            return new ChatAnthropic({
                apiKey: getApiKey(provider),
                model: getDefaultModel(provider),
                temperature,
                maxTokens,
            });
        }
        if (provider === 'openai') {
            return new ChatOpenAI({
                apiKey: getApiKey(provider),
                model: getDefaultModel(provider),
                temperature,
                maxTokens,
            });
        }
        if (provider === 'google') {
            return new ChatGoogleGenerativeAI({
                apiKey: getApiKey(provider),
                model: getDefaultModel(provider),
                temperature,
                maxTokens,
            });
        }
        if (provider === 'xai') {
            return new ChatOpenAI({
                apiKey: getApiKey(provider),
                configuration: { baseURL: 'https://api.x.ai/v1' },
                model: getDefaultModel(provider),
                temperature,
                maxTokens,
            });
        }
        throw new Error(`Unsupported provider: ${provider}`);
    }

    _resolveProvider(provider) {
        const available = this.getAvailableProviders();
        if (provider && available.includes(provider)) {
            return provider;
        }
        return available.length > 0 ? available[0] : null;
    }

    _buildMessages(prompt) {
        return [
            new SystemMessage('You are a helpful AI assistant.'),
            new HumanMessage(prompt),
        ];
    }

    _extractText(provider, result) {
        if (provider === 'google' && typeof result?.text !== 'undefined') {
            return String(result.text);
        }
        return String(result?.content ?? '');
    }

    async askQuestion(topic, provider = null, template = '{topic}', maxTokens = 1000, temperature = 0.7) {
        const prompt = template.replace('{topic}', topic);
        const resolvedProvider = this._resolveProvider(provider);

        if (!resolvedProvider) {
            return {
                success: false,
                error: 'No providers available',
                provider: 'none',
                model: 'none',
                prompt,
                response: null,
            };
        }

        const model = getDefaultModel(resolvedProvider);

        try {
            const client = this._createClient(resolvedProvider, temperature, maxTokens);
            const result = await client.invoke(this._buildMessages(prompt));
            return {
                success: true,
                provider: resolvedProvider,
                model,
                prompt,
                response: this._extractText(resolvedProvider, result),
                temperature,
                maxTokens,
            };
        } catch (error) {
            return {
                success: false,
                provider: resolvedProvider,
                model,
                prompt,
                error: error.message,
                response: null,
                temperature,
                maxTokens,
            };
        }
    }
}

async function main() {
    const args = process.argv.slice(2);

    if (args.length > 0 && args[0] === 'web') {
        try {
            const { runWebServer } = await import('../web.js');
            await runWebServer(LangChainLLMManager);
        } catch (error) {
            console.error('Error: web.js not found or Express not installed.');
            console.error('Install Express: npm install express cors');
            process.exit(1);
        }
    } else {
        const manager = new LangChainLLMManager();
        await manager._checkProviders();
        await interactiveCli(manager);
    }
}

export { LangChainLLMManager };

if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(console.error);
}
